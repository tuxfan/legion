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
        
        
        
        class ImageReduction {
            
        private:
            
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
            
            
            template<int layerID>
            class CompositeProjectionFunctor : public ProjectionFunctor {
            public:
                CompositeProjectionFunctor(void){
                    mFunctorID = -1;
                }
                CompositeProjectionFunctor(int functorID){
                    mFunctorID = functorID;
                }
                
                virtual LogicalRegion project(Context context, Task *task,
                                              unsigned index,
                                              LogicalPartition upperBound,
                                              const DomainPoint &point) {
                    CompositeArguments args = ((CompositeArguments*)(task->local_args))[0];
                    DomainPoint newPoint = point;
                    newPoint[2] = (layerID == 0) ? args.layer0 : args.layer1;
                    LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(context, upperBound, newPoint);
                    return result;
                }
                
                virtual bool is_exclusive(void) const{ return true; }
                virtual unsigned get_depth(void) const{ return 0; }
                int functorID() const{ return mFunctorID; }
                
            private:
                int mFunctorID;
            };
            
            
            
        public:
            
            ImageReduction(){}
            ImageReduction(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
            virtual ~ImageReduction();
            
            FutureMap launch_task_by_depth(unsigned taskID);
            FutureMap reduce_associative_commutative();
            FutureMap reduce_associative_noncommutative(int ordering[]);
            FutureMap reduce_nonassociative_commutative();
            FutureMap reduce_nonassociative_noncommutative(int ordering[]);
            Future display(int t);
            
            void set_blend_func(GLenum sfactor, GLenum dfactor) {
                mBlendFunctionSource = sfactor;
                mBlendFunctionDestination = dfactor;
            }
            
            void set_depth_func(GLenum func){ mDepthFunction = func; }
            
            static void create_image_field_pointers(ImageSize imageSize,
                                                    PhysicalRegion region,
                                                    int layer,
                                                    PixelField *&r,
                                                    PixelField *&g,
                                                    PixelField *&b,
                                                    PixelField *&a,
                                                    PixelField *&z,
                                                    PixelField *&userdata,
                                                    ByteOffset stride[3]);
            
            static std::string describe_task(const Task *task) {
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
            
            static void display_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime);
            
            static PhysicalRegion compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, PhysicalRegion region1);
            
            static void composite_task(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime *runtime);
            
            
            FieldSpace imageFields();
            void createImage();
            void partitionImageByDepth();
            void prepareCompositePartition();
            void prepareCompositeDomains();
            void prepareCompositeLaunchDomains();
            Domain compositeDomain(int level);
            void prepareProjectionFunctors();
            void prepareImageForComposite();
            FutureMap reduceAssociative(int permutation[]);
            FutureMap reduceNonassociative(int permutation[]);
            FutureMap launchCompositeTaskTree(int permutation[]);
            FutureMap launchCompositeTaskPipeline(int permutation[]);
            FutureMap launchTreeLevel(int level, int permutation[]);
            void addCompositeArgumentsToArgmap(CompositeArguments *args, int taskZ, ArgumentMap &argMap);
            void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);
            int *defaultPermutation();
            void addImageFieldsToRequirement(RegionRequirement &req);
            void registerTasks();
            int numTreeLevels();
            
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
            vector<Domain> mCompositeTreeDomain;
            GLenum mDepthFunction;
            GLenum mBlendFunctionSource;
            GLenum mBlendFunctionDestination;
            CompositeProjectionFunctor<0> *mFunctor0;
            CompositeProjectionFunctor<1> *mFunctor1;
        };
        
    }
}

#endif /* ImageReduction_h */
