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
        
        typedef std::vector<Future> Futures;

        typedef struct {
            LogicalRegion region;
            int layer;
        } CompositeResult;

        typedef struct {
            ImageSize imageSize;
            int layer0;
            int layer1;
        } CompositeArguments;

        typedef struct {
            ImageSize imageSize;
            int t;
        } DisplayArguments;

        
        class RenderSpace {
        public:
            
            RenderSpace(){}
            RenderSpace(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
            virtual ~RenderSpace();
            void prepareToDestruct();
            
            FutureMap launchTaskByDepth(unsigned taskID);
            Futures reduceAssociativeCommutative();
            Futures reduceAssociativeNoncommutative(int ordering[]);
            Futures reduceNonassociativeCommutative();
            Futures reduceNonassociativeNoncommutative(int ordering[]);
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
            
            static CompositeResult composite_leaf_task(const Task *task,
                                                     const std::vector<PhysicalRegion> &regions,
                                                     Context ctx, Runtime *runtime);
            
            static CompositeResult composite_internal_task(const Task *task,
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
            
            static void registerTasks();
            
            
        private:
            void nameTasks();
            FieldSpace imageFields();
            void createImage();
            void partitionImageByDepth();
            void partitionImageForComposite();
            void addCompositeRegionRequirement(Point<DIMENSIONS> point, int depth, TaskLauncher &taskLauncher);
            void addFutureToLauncher(TaskLauncher &taskLauncher, Future future);
            Future launchCompositeTask(Point<DIMENSIONS> point, int depth0, int depth1);
            Future launchCompositeTask(Future future0, Future future1);
            Futures launchCompositeTaskTreeLevel( Futures futures);
            Futures launchCompositeTaskTreeLeaves(int ordering[]);
            Futures launchCompositeTaskTree(int ordering[]);
            Futures launchCompositeTaskPipeline(int ordering[]);
            Futures reduceAssociative(int permutation[]);
            Futures reduceNonassociative(int permutation[]);
            int *defaultPermutation();
            void addImageFieldsToRequirement(RegionRequirement &req);
            LogicalRegion createDisplayPlane();
            
            static PhysicalRegion compositeTwoRegions(ImageSize imageSize, PhysicalRegion region0, int layer0, PhysicalRegion region1, int layer1);
            
            static void compositeTwoPixels(PixelField *r0,
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
                                           PixelField *userdataOut);
            
            static void createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                Rect<3> imageBounds, PhysicalRegion region, ByteOffset offset[]);
            
            ImageSize mImageSize;
            Context mContext;
            Runtime *mRuntime;
            LogicalRegion mImage;
            Domain mImageDomain;
            Domain mDepthDomain;
            Domain mCompositeDomain;
            Domain mDisplayDomain;
            LogicalPartition mDepthPartition;
            LogicalPartition mCompositePartition;
            int *mDefaultPermutation;
        };
        
        
        enum TaskIDs {
            COMPOSITE_LEAF_TASK_ID = MAX_APPLICATION_TASK_ID - 3,//problem, how to reserve these task ids
            COMPOSITE_INTERNAL_TASK_ID,
            DISPLAY_TASK_ID,
        };
        
    }
}

#endif /* RenderSpace_h */
