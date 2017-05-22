/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef ImageReduction_h
#define ImageReduction_h


#include "legion.h"
#include "legion_visualization.h"
#include "image_reduction_composite.h"

#include "usec_timer.h"

#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>

#include <iostream>
#include <sstream>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

namespace Legion {
    namespace Visualization {
        
        typedef struct {
            ImageSize imageSize;
            int layer0;
            int layer1;
            GLenum depthFunction;
            GLenum blendFunctionSource;
            GLenum blendFunctionDestination;
        } CompositeArguments;
        
        
        
        typedef struct {
            ImageSize imageSize;
            int t;
        } DisplayArguments;
        
        
        class ImageReduction {
            
        public:
            class CompositeProjectionFunctor : public ProjectionFunctor {
            public:
                CompositeProjectionFunctor(Runtime *runtime){ mRuntime = runtime; }
                
                virtual LogicalRegion project(Context context, Task *task,
                                              unsigned index,
                                              LogicalPartition upperBound,
                                              const DomainPoint &point) {
                    LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(context, upperBound, newPoint(point));
                    return result;
                }
                
                virtual bool is_exclusive(void) const{ return true; }
                virtual unsigned get_depth(void) const{ return 0; }
                
            protected:
                virtual DomainPoint newPoint(DomainPoint point) = 0;
                Runtime *mRuntime;
            };
            
            // ProjectionFunctor has to be a pure function (currently)
            // So make this a template that depends on increment
            // In the future this also has to depend on ordering which is a dynamic int array
            
            template<int increment>
            class CompositeProjectionFunctorClass : public CompositeProjectionFunctor {
            public:
                CompositeProjectionFunctorClass(void){}
            private:
                virtual DomainPoint newPoint(DomainPoint point) {
                    Point<DIMENSIONS> p = point.get_point<DIMENSIONS>();
                    int layer = p.x[2] * NUM_FRAGMENTS_PER_COMPOSITE_TASK + increment;
                    //p.x[2] = mOrdering[layer];//no ordering yet, commutative only
                    p.x[2] = layer;
                    DomainPoint result = DomainPoint::from_point<DIMENSIONS>(p);
                    return result;
                }
            };
            
            
            
            
            ImageReduction(){}
            ImageReduction(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
            virtual ~ImageReduction();
            
            FutureMap launchTaskByDepth(unsigned taskID);
            FutureMap reduceAssociativeCommutative();
            FutureMap reduceAssociativeNoncommutative(int ordering[]);
            FutureMap reduceNonassociativeCommutative();
            FutureMap reduceNonassociativeNoncommutative(int ordering[]);
            Future display(int t);
            void setBlendFunc(GLenum sfactor, GLenum dfactor) {
                mBlendFunctionSource = sfactor;
                mBlendFunctionDestination = dfactor;
            }
            void setDepthFunc(GLenum func){ mDepthFunction = func; }
            
            static void createImageFieldPointers(ImageSize imageSize,
                                                 PhysicalRegion region,
                                                 int layer,
                                                 PixelField *&r,
                                                 PixelField *&g,
                                                 PixelField *&b,
                                                 PixelField *&a,
                                                 PixelField *&z,
                                                 PixelField *&userdata,
                                                 ByteOffset stride[3]);
            
            
            static void display_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime);
            
            static PhysicalRegion compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, PhysicalRegion region1);
            
            static void composite_task(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime *runtime);
            
            
            static std::string describeTask(const Task *task) {
                std::ostringstream output;
                output << task->get_task_name() << " " << task->get_unique_id()
                << " (" << task->index_point.point_data[0]
                << ", " << task->index_point.point_data[1]
                << ", " << task->index_point.point_data[2]
                << ")"
                ;
                return output.str();
            }
            
            
            
        private:
            typedef struct {
                CompositeProjectionFunctor* functor0;
                CompositeProjectionFunctor* functor1;
                int functorID0;
                int functorID1;
                Domain domain;
            } CompositeLaunchDescriptor;
            
            FieldSpace imageFields();
            void createImage();
            void partitionImageByDepth();
            void prepareCompositePartition();
            void prepareCompositeLaunchDomains();
            CompositeProjectionFunctor* newProjectionFunctor(int increment);
            Domain compositeDomain(int level);
            void partitionImageForComposite();
            void prepareProjectionFunctors();
            FutureMap reduceAssociative(int permutation[]);
            FutureMap reduceNonassociative(int permutation[]);
            FutureMap launchCompositeTaskTree(int permutation[]);
            FutureMap launchCompositeTaskPipeline(int permutation[]);
            FutureMap launchTreeLevel(int level, int permutation[]);
            void addCompositeArgumentsToArgmap(CompositeArguments *args, int taskZ, ArgumentMap &argMap);
            void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int level, bool isLeft, PrivilegeMode privilege, CoherenceProperty coherence);
            int *defaultPermutation();
            void addImageFieldsToRequirement(RegionRequirement &req);
            void registerTasks();
            
            
            static void createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                Rect<DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]);
            
            ImageSize mImageSize;
            Context mContext;
            Runtime *mRuntime;
            LogicalRegion mImage;
            Domain mImageDomain;
            Domain mDepthDomain;
            Domain mCompositePipelineDomain;
            Domain mDisplayDomain;
            LogicalPartition mDepthPartition;
            LogicalPartition mCompositePartition;
            int *mDefaultPermutation;
            TaskID mCompositeTaskID;
            TaskID mDisplayTaskID;
            vector<CompositeLaunchDescriptor> mCompositeLaunchDescriptor;
            GLenum mDepthFunction;
            GLenum mBlendFunctionSource;
            GLenum mBlendFunctionDestination;
        };
        
    }
}

#endif /* ImageReduction_h */
