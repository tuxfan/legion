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

#include "RenderSpace.h"

#include <iostream>
#include <fstream>
#include <math.h>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
    namespace Visualization {
        
        RenderSpace::RenderSpace(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime) {
            mImageSize = imageSize;
            assert(imageSize.depth % 2 == 0);
            mContext = ctx;
            mRuntime = runtime;
            mDefaultPermutation = nullptr;
            initializeTimers();
            createImage();
            partitionImageByDepth();
            partitionImageForComposite();
            nameTasks();
        }
        
        RenderSpace::~RenderSpace() {
            if(mDefaultPermutation != nullptr) {
                delete [] mDefaultPermutation;
                mDefaultPermutation = nullptr;
            }
        }
        
        
        
        void RenderSpace::registerTasks() {
            
            HighLevelRuntime::register_legion_task<display_task>(DISPLAY_TASK_ID,
                                                                 Processor::LOC_PROC, false/*single*/, true/*index*/);
            
            HighLevelRuntime::register_legion_task<CompositeResult, composite_leaf_task>(COMPOSITE_LEAF_TASK_ID,
                                                                                       Processor::LOC_PROC, true/*single*/, false/*index*/);
            
            HighLevelRuntime::register_legion_task<CompositeResult, composite_internal_task>(COMPOSITE_INTERNAL_TASK_ID,
                                                                                           Processor::LOC_PROC, true/*single*/, false/*index*/);
        }
        
        
        void RenderSpace::nameTasks() {
            mRuntime->attach_name(DISPLAY_TASK_ID, "display task");
            mRuntime->attach_name(COMPOSITE_LEAF_TASK_ID, "composite leaf task");
            mRuntime->attach_name(COMPOSITE_INTERNAL_TASK_ID, "composite internal task");
        }
        
        void RenderSpace::initializeTimers() {
            mCompositeLaunchTimer = new UsecTimer("time launching composite tasks:");
            mCompositeTaskCount = 0;
        }
        
        
        void RenderSpace::reportTimers() {
            cout << "launched " << mCompositeTaskCount << " composite tasks" << endl;
            cout << mCompositeLaunchTimer->to_string() << endl;
        }
        
        
        FieldSpace RenderSpace::imageFields() {
            FieldSpace fields = mRuntime->create_field_space(mContext);
            mRuntime->attach_name(fields, "pixel fields");
            {
                FieldAllocator allocator = mRuntime->create_field_allocator(mContext, fields);
                FieldID fidr = allocator.allocate_field(sizeof(PixelField), FID_FIELD_R);
                assert(fidr == FID_FIELD_R);
                FieldID fidg = allocator.allocate_field(sizeof(PixelField), FID_FIELD_G);
                assert(fidg == FID_FIELD_G);
                FieldID fidb = allocator.allocate_field(sizeof(PixelField), FID_FIELD_B);
                assert(fidb == FID_FIELD_B);
                FieldID fida = allocator.allocate_field(sizeof(PixelField), FID_FIELD_A);
                assert(fida == FID_FIELD_A);
                FieldID fidz = allocator.allocate_field(sizeof(PixelField), FID_FIELD_Z);
                assert(fidz == FID_FIELD_Z);
                FieldID fidUserdata = allocator.allocate_field(sizeof(PixelField), FID_FIELD_USERDATA);
                assert(fidUserdata == FID_FIELD_USERDATA);
            }
            return fields;
        }
        
        
        void RenderSpace::createImage() {
            Rect<DIMENSIONS> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<DIMENSIONS>::ONES());
            mImageDomain = Domain::from_rect<DIMENSIONS>(imageBounds);
            //TODO mRuntime->attach_name(mImageDomain, "image domain");
            IndexSpace pixels = mRuntime->create_index_space(mContext, mImageDomain);
            mRuntime->attach_name(pixels, "image index space");
            FieldSpace fields = imageFields();
            mImage = mRuntime->create_logical_region(mContext, pixels, fields);
            mRuntime->attach_name(mImage, "image");
        }
        
        
        void RenderSpace::partitionImageByDepth() {
            Blockify<DIMENSIONS> coloring(mImageSize.layerSize());
            IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageDepthIndexPartition, "image depth partition");
            mDepthPartition = mRuntime->get_logical_partition(mContext, mImage, imageDepthIndexPartition);
            Rect<DIMENSIONS> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<DIMENSIONS>::ONES());
            mDepthDomain = Domain::from_rect<DIMENSIONS>(depthBounds);
            //TODO mRuntime->attach_name(mDepthDomain, "depth domain");
        }
        
        
        void RenderSpace::partitionImageForComposite() {
            Blockify<DIMENSIONS> coloring(mImageSize.fragmentSize());
            IndexPartition imageCompositeIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageCompositeIndexPartition, "image composite partition");
            mCompositePartition = mRuntime->get_logical_partition(mContext, mImage, imageCompositeIndexPartition);
            Rect<DIMENSIONS> compositeBounds(mImageSize.origin(), mImageSize.numFragments() - Point<DIMENSIONS>::ONES());
            mCompositeDomain = Domain::from_rect<DIMENSIONS>(compositeBounds);
            //TODO mRuntime->attach_name(mCompositeDomain, "composite domain");
        }
        
        
        void RenderSpace::addImageFieldsToRequirement(RegionRequirement &req) {
            req.add_field(FID_FIELD_R);
            req.add_field(FID_FIELD_G);
            req.add_field(FID_FIELD_B);
            req.add_field(FID_FIELD_A);
            req.add_field(FID_FIELD_Z);
            req.add_field(FID_FIELD_USERDATA);
        }
        
        
        void RenderSpace::createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                  Rect<DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]) {
            acc = region.get_field_accessor(fieldID).typeify<PixelField>();
            Rect<DIMENSIONS> tempBounds;
            field = acc.raw_rect_ptr<DIMENSIONS>(imageBounds, tempBounds, offset);
            assert(imageBounds == tempBounds);
        }
        
        
        void RenderSpace::createImageFieldPointers(ImageSize imageSize,
                                                   PhysicalRegion region,
                                                   int layer,
                                                   PixelField *&r,
                                                   PixelField *&g,
                                                   PixelField *&b,
                                                   PixelField *&a,
                                                   PixelField *&z,
                                                   PixelField *&userdata,
                                                   ByteOffset stride[3]) {
            
            Rect<DIMENSIONS> tempBounds;
            Point<3> origin = imageSize.origin();
            origin.x[2] = layer;
            Point<3> upperBound = imageSize.upperBound() - Point<3>::ONES();
            upperBound.x[2] = layer;
            Rect<DIMENSIONS> imageBounds = Rect<DIMENSIONS>(origin, upperBound);
            
            RegionAccessor<AccessorType::Generic, PixelField> acc_r, acc_g, acc_b, acc_a, acc_z, acc_userdata;
            
            createImageFieldPointer(acc_r, FID_FIELD_R, r, imageBounds, region, stride);
            createImageFieldPointer(acc_g, FID_FIELD_G, g, imageBounds, region, stride);
            createImageFieldPointer(acc_b, FID_FIELD_B, b, imageBounds, region, stride);
            createImageFieldPointer(acc_a, FID_FIELD_A, a, imageBounds, region, stride);
            createImageFieldPointer(acc_z, FID_FIELD_Z, z, imageBounds, region, stride);
            createImageFieldPointer(acc_userdata, FID_FIELD_USERDATA, userdata, imageBounds, region, stride);
        }
        
        
        FutureMap RenderSpace::launchTaskByDepth(unsigned taskID){
            ArgumentMap argMap;
            IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(&mImageSize, sizeof(ImageSize)), argMap);
            //TODO mRuntime->attach_name(depthLauncher, "depth task launcher");
            RegionRequirement req(mDepthPartition, 0, READ_WRITE, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            depthLauncher.add_region_requirement(req);
            FutureMap futures = mRuntime->execute_index_space(mContext, depthLauncher);
            return futures;
        }
        
        
        
        int *RenderSpace::defaultPermutation(){
            if(mDefaultPermutation == nullptr) {
                mDefaultPermutation = new int[mImageSize.depth];
                for(int i = 0; i < mImageSize.depth; ++i) {
                    mDefaultPermutation[i] = i;
                }
            }
            return mDefaultPermutation;
        }
        
        
        inline void RenderSpace::compositeTwoPixels(PixelField *r0,
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
                                                    PixelField *userdataOut) {
            
            // hidden surface elimination
            if(*z0 < *z1) {
                *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
            } else {
                *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
            }
            
        }
        
        
        inline PhysicalRegion RenderSpace::compositeTwoRegions(ImageSize imageSize, PhysicalRegion region0, int layer0, PhysicalRegion region1, int layer1) {
            
            ByteOffset stride[DIMENSIONS];
            PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
            createImageFieldPointers(imageSize, region0, layer0, r0, g0, b0, a0, z0, userdata0, stride);
            PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
            createImageFieldPointers(imageSize, region1, layer1, r1, g1, b1, a1, z1, userdata1, stride);

#pragma unroll
            for(int i = 0; i < imageSize.numPixelsPerFragment(); ++i) {
                compositeTwoPixels(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0);
                r0++; g0++; b0++; a0++; z0++; userdata0++;
                r1++; g1++; b1++; a1++; z1++; userdata1++;
            }
            return region0;
        }
        
        
        
        
        CompositeResult RenderSpace::composite_leaf_task(const Task *task,
                                                       const std::vector<PhysicalRegion> &regions,
                                                       Context ctx, HighLevelRuntime *runtime) {
            
            UsecTimer composite(describeTask(task) + " leaf:");
            composite.start();
            PhysicalRegion fragment0 = regions[0];
            PhysicalRegion fragment1 = regions[1];
            CompositeArguments args = ((CompositeArguments*)task->args)[0];

#ifdef NULL_COMPOSITE_TASKS
            return (CompositeResult) { fragment0.get_logical_region(), args.layer0 };//performance testing
#endif

            PhysicalRegion compositedResult = compositeTwoRegions(args.imageSize, fragment0, args.layer0, fragment1, args.layer1);
            composite.stop();
            //cout << composite.to_string() << endl;
            CompositeResult result = { compositedResult.get_logical_region(), args.layer0 };
            return result;
        }
        
        
        CompositeResult RenderSpace::composite_internal_task(const Task *task,
                                                           const std::vector<PhysicalRegion> &regions,
                                                           Context ctx, HighLevelRuntime *runtime) {
            
            UsecTimer composite(describeTask(task) + " internal:");
            composite.start();
            PhysicalRegion fragment0 = regions[0];
            PhysicalRegion fragment1 = regions[1];
            CompositeArguments args = ((CompositeArguments*)task->args)[0];

#ifdef NULL_COMPOSITE_TASKS
            return (CompositeResult) { fragment0.get_logical_region(), args.layer0 };//performance testing
#endif

            PhysicalRegion compositedResult = compositeTwoRegions(args.imageSize, fragment0, args.layer0, fragment1, args.layer1);
            composite.stop();
            //cout << composite.to_string() << endl;
            CompositeResult result = { compositedResult.get_logical_region(), args.layer0 };
            return result;
        }
        
        
        void RenderSpace::addCompositeRegionRequirement(Point<DIMENSIONS> point, int depth, TaskLauncher &taskLauncher) {
            DomainPoint domainPoint;
            domainPoint.dim = DIMENSIONS;
            for(int i = 0; i < DIMENSIONS; ++i) {
                domainPoint[i] = point.x[i];
            }
            domainPoint[2] = depth;
            LogicalRegion region = mRuntime->get_logical_subregion_by_color(mCompositePartition, domainPoint);
            RegionRequirement req(region, READ_ONLY, SIMULTANEOUS, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
        }
        
        
        Future RenderSpace::launchCompositeTask(Point<DIMENSIONS> point, int depth0, int depth1) {
            CompositeArguments args = { mImageSize, depth0, depth1 };
            TaskLauncher taskLauncher(COMPOSITE_LEAF_TASK_ID, TaskArgument(&args, sizeof(args)));
            addCompositeRegionRequirement(point, depth0, taskLauncher);
            addCompositeRegionRequirement(point, depth1, taskLauncher);
            mCompositeTaskCount++;
            Future resultFuture = mRuntime->execute_task(mContext, taskLauncher);
            return resultFuture;
        }
        
        
        void RenderSpace::addFutureToLauncher(TaskLauncher &taskLauncher, Future future) {
            taskLauncher.add_future(future);
            LogicalRegion region = future.get_result<LogicalRegion>();
            RegionRequirement req(region, READ_ONLY, SIMULTANEOUS, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
        }
        
        
        Future RenderSpace::launchCompositeTask(Future future0, Future future1) {
            CompositeResult result0 = future0.get_result<CompositeResult>();
            CompositeResult result1 = future1.get_result<CompositeResult>();
            CompositeArguments args = { mImageSize, result0.layer, result1.layer };
            TaskLauncher taskLauncher(COMPOSITE_INTERNAL_TASK_ID, TaskArgument(&args, sizeof(args)));
            addFutureToLauncher(taskLauncher, future0);
            addFutureToLauncher(taskLauncher, future1);
            mCompositeTaskCount++;
            Future resultFuture = mRuntime->execute_task(mContext, taskLauncher);
            return resultFuture;
        }
        
        
        Future RenderSpace::launchCompositeTask(Point<DIMENSIONS> point, Future priorResult, int nextInput) {
            CompositeResult result = priorResult.get_result<CompositeResult>();
            CompositeArguments args = { mImageSize, result.layer, nextInput };
            TaskLauncher taskLauncher(COMPOSITE_INTERNAL_TASK_ID, TaskArgument(&args, sizeof(args)));
            addFutureToLauncher(taskLauncher, priorResult);
            addCompositeRegionRequirement(point, nextInput, taskLauncher);
            mCompositeTaskCount++;
            Future resultFuture = mRuntime->execute_task(mContext, taskLauncher);
            return resultFuture;
        }
        
        
        Futures RenderSpace::launchCompositeTaskTreeLevel(Futures futures) {
            Futures result = Futures();
            for(int i = 0; i < futures.size(); i += 2) {
                Future future0 = futures[i];
                Future future1 = futures[i + 1];
                result.push_back(launchCompositeTask(future0, future1));
            }
            return (result.size() == 1 ? result : launchCompositeTaskTreeLevel(result));
        }
        
        
        Futures RenderSpace::launchCompositeTaskTreeLeaves(int ordering[]) {
            Futures futures = Futures();
            for(int order = 0; order < mImageSize.depth; order += NUM_FRAGMENTS_PER_COMPOSITE_TASK) {
                Point<DIMENSIONS> point = Point<DIMENSIONS>::ZEROES();
                point.x[2] = order;
                for(int fragment = 0; fragment < mImageSize.numFragmentsPerLayer; ++fragment) {
                    futures.push_back(launchCompositeTask(point, ordering[order], ordering[order + 1]));
                    point = mImageSize.incrementFragment(point);
                }
            }
            return futures;
        }
        
        
        Futures RenderSpace::launchCompositeTaskTree(int ordering[]) {
            mCompositeLaunchTimer->start();
            Futures futures = launchCompositeTaskTreeLeaves(ordering);
            mCompositeLaunchTimer->stop();
            return launchCompositeTaskTreeLevel(futures);
        }
        
        
        Futures RenderSpace::launchCompositeTaskPipeline(int ordering[]) {
            Futures result;
            return result;
        }
        
        
        Futures RenderSpace::reduceAssociative(int permutation[]) {
            return launchCompositeTaskTree(permutation);
        }
        
        Futures RenderSpace::reduceAssociativeCommutative(){
            return reduceAssociative(defaultPermutation());
        }
        
        Futures RenderSpace::reduceAssociativeNoncommutative(int ordering[]){
            return reduceAssociative(ordering);
        }
        
        Futures RenderSpace::reduceNonassociative(int ordering[]) {
            mCompositeLaunchTimer->start();
            Futures futures = Futures();
            Point<DIMENSIONS> point = Point<DIMENSIONS>::ZEROES();
            for(int fragment = 0; fragment < mImageSize.numFragmentsPerLayer; ++fragment) {
                Future future = launchCompositeTask(point, ordering[0], ordering[1]);
                for(int order = 2; order < mImageSize.depth; ++order) {
                    future = launchCompositeTask(point, future, ordering[order]);
                }
                futures.push_back(future);
                point = mImageSize.incrementFragment(point);
            }
            mCompositeLaunchTimer->stop();
            return futures;
        }
        
        Futures RenderSpace::reduceNonassociativeCommutative(){
            return reduceNonassociative(defaultPermutation());
        }
        
        Futures RenderSpace::reduceNonassociativeNoncommutative(int ordering[]){
            return reduceNonassociative(ordering);
        }
        
        
        
        void RenderSpace::display_task(const Task *task,
                                                       const std::vector<PhysicalRegion> &regions,
                                                       Context ctx, HighLevelRuntime *runtime) {
            
            DisplayArguments args = ((DisplayArguments*)task->args)[0];
            string outputFileName = "display." + ::to_string(args.t) + ".txt";
            UsecTimer display(describeTask(task) + " write to " + outputFileName + ":");
            display.start();
            PhysicalRegion displayPlane = regions[0];
            ByteOffset stride[DIMENSIONS];
            PixelField *r, *g, *b, *a, *z, *userdata;
            createImageFieldPointers(args.imageSize, displayPlane, args.imageSize.depth - 1, r, g, b, a, z, userdata, stride);
            
            FILE *outputFile = fopen(outputFileName.c_str(), "wb");
            fwrite(r, 6 * sizeof(*r), args.imageSize.pixelsPerLayer(), outputFile);
            fclose(outputFile);

            display.stop();
            cout << display.to_string() << endl;
        }
        
        
        LogicalRegion RenderSpace::createDisplayPlane() {
            DomainPoint point;
            point.dim = DIMENSIONS;
            point[0] = 0;
            point[1] = 0;
            point[2] = mImageSize.depth - 1;
            LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mDepthPartition, point);
            return displayPlane;
        }

        
        Future RenderSpace::display(int t) {
            DisplayArguments args = { mImageSize, t };
            TaskLauncher taskLauncher(DISPLAY_TASK_ID, TaskArgument(&args, sizeof(args)));
            LogicalRegion displayPlane = createDisplayPlane();
            RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
            Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
            return displayFuture;
        }
        
    }
}
