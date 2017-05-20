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


#ifndef RenderSpace_h
#define RenderSpace_h

#include "legion.h"
#include "legion_visualization.h"

#include "UsecTimer.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

namespace Legion {
    namespace Visualization {
        
        typedef struct {
            ImageSize imageSize;
            int layer0;
            int layer1;
            int depthFunction;
            int blendFunction;
        } CompositeArguments;
        
        typedef struct {
            ImageSize imageSize;
            int t;
        } DisplayArguments;
        
        
        class RenderSpace {
            
        public:
            class CompositeProjectionFunctor : public ProjectionFunctor {
            public:
                CompositeProjectionFunctor(void){}
                
                virtual LogicalRegion project(Context context, Task *task,
                                              unsigned index,
                                              LogicalPartition upperBound,
                                              const DomainPoint &point) {
                    LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(context, upperBound, newPoint(point));
                    return result;
                }
                
                virtual LogicalRegion project(Context context, Task *task,
                                              unsigned index,
                                              LogicalRegion upperBound,
                                              const DomainPoint &point) { return LogicalRegion::NO_REGION; }
                virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                              LogicalRegion upperBound,
                                              const DomainPoint &point) { return LogicalRegion::NO_REGION; }
                virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                              LogicalPartition upperBound,
                                              const DomainPoint &point) { return LogicalRegion::NO_REGION; }
                virtual bool is_exclusive(void) const{ return true; }
                virtual unsigned get_depth(void) const{ return 0; }
                
            protected:
                virtual DomainPoint newPoint(DomainPoint point) = 0;
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
                    
                    cout << "functor task at " << point << " increment " << increment << " access subregion " << result << endl;
                    
                    return result;
                }
            };
            
            
            
            
        public:
            
            RenderSpace(){}
            RenderSpace(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
            virtual ~RenderSpace();
            
            FutureMap launchTaskByDepth(unsigned taskID);
            FutureMap reduceAssociativeCommutative();
            FutureMap reduceAssociativeNoncommutative(int ordering[]);
            FutureMap reduceNonassociativeCommutative();
            FutureMap reduceNonassociativeNoncommutative(int ordering[]);
            Future display(int t);
            
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
            
            static void composite_task(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, Runtime *runtime);
            
            
            static std::string describeTask(const Task *task) {
                return std::string(task->get_task_name()) + " "
                + std::to_string(task->get_unique_id()) +
                + " ("
                + std::to_string(task->index_point.point_data[0]) + ","
                + std::to_string(task->index_point.point_data[1]) + ","
                + std::to_string(task->index_point.point_data[2]) + ")";
            }
            
            
            
        private:
            FieldSpace imageFields();
            void createImage();
            void partitionImageByDepth();
            void fragmentImageLayers();
            void prepareCompositeLaunchDomains();
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
            LogicalRegion createDisplayPlane();
            void registerTasks();
            
            typedef void(*CompositeFunction)
            (PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*,
            PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*,
            PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, int);
            
            static CompositeFunction compositeFunctionPointer(int depthFunction, int blendFunction);
            
            static PhysicalRegion compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, PhysicalRegion region1);
            
            static void compositePixelsLess(PixelField *r0,
                                            PixelField *g0,
                                            PixelField *b0,
                                            PixelField *a0,
                                            PixelField *z0,
                                            PixelField *userdata0,
                                            PixelField *r1,
                                            PixelField *g1,
                                            PixelField *b1,
                                            PixelField *a1,
                                            PixelField *z1,
                                            PixelField *userdata1,
                                            PixelField *rOut,
                                            PixelField *gOut,
                                            PixelField *bOut,
                                            PixelField *aOut,
                                            PixelField *zOut,
                                            PixelField *userdataOut,
                                            int numPixels);
            
            static void createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                Rect<DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]);
            
            ImageSize mImageSize;
            Context mContext;
            Runtime *mRuntime;
            LogicalRegion mImage;
            Domain mImageDomain;
            Domain mDepthDomain;
            Domain mCompositeTreeDomain;
            Domain mCompositePipelineDomain;
            Domain mDisplayDomain;
            LogicalPartition mDepthPartition;
            LogicalPartition mCompositePartition;
            int *mDefaultPermutation;
            TaskID mCompositeTaskID;
            TaskID mDisplayTaskID;
            vector<CompositeProjectionFunctor*> *mProjectionFunctors;
            
        };
        
    }
}

#endif /* RenderSpace_h */
