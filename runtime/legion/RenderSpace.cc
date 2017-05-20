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

#include <OpenGL/OpenGL.h>

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
            createImage();
            partitionImageByDepth();
            partitionImageForComposite();
            registerTasks();
        }
        
        RenderSpace::~RenderSpace() {
            if(mDefaultPermutation != nullptr) {
                delete [] mDefaultPermutation;
                mDefaultPermutation = nullptr;
            }
        }
        
        
        
        void RenderSpace::registerTasks() {
            
            LayoutConstraintRegistrar layoutRegistrar(imageFields(), "layout");
            LayoutConstraintID layoutConstraintID = mRuntime->register_layout(layoutRegistrar);
            
            mCompositeTaskID = mRuntime->generate_dynamic_task_id();
            TaskVariantRegistrar compositeRegistrar(mCompositeTaskID, "compositeTask");
            compositeRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
            .add_layout_constraint_set(0/*index*/, layoutConstraintID);
            mRuntime->register_task_variant<composite_task>(compositeRegistrar);
            mRuntime->attach_name(mCompositeTaskID, "compositeTask");
            
            mDisplayTaskID = mRuntime->generate_dynamic_task_id();
            TaskVariantRegistrar displayRegistrar(mDisplayTaskID, "displayTask");
            displayRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
            .add_layout_constraint_set(0/*index*/, layoutConstraintID);
            mRuntime->register_task_variant<display_task>(displayRegistrar);
            mRuntime->attach_name(mDisplayTaskID, "displayTask");
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
        
        
        void RenderSpace::fragmentImageLayers() {
            Blockify<DIMENSIONS> coloring(mImageSize.fragmentSize());
            IndexPartition imageCompositeIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageCompositeIndexPartition, "image composite partition");
            mCompositePartition = mRuntime->get_logical_partition(mContext, mImage, imageCompositeIndexPartition);
        }
        
        
        void RenderSpace::prepareCompositeLaunchDomains() {
            Point<DIMENSIONS> numTreeComposites = mImageSize.numFragments();
            numTreeComposites.x[2] /= NUM_FRAGMENTS_PER_COMPOSITE_TASK;
            Rect<DIMENSIONS> compositeTreeBounds(mImageSize.origin(), numTreeComposites - Point<DIMENSIONS>::ONES());
            mCompositeTreeDomain = Domain::from_rect<DIMENSIONS>(compositeTreeBounds);
            Rect<DIMENSIONS> compositePipelineBounds(mImageSize.origin(), mImageSize.numFragments() - Point<DIMENSIONS>::ONES());
            mCompositePipelineDomain = Domain::from_rect<DIMENSIONS>(compositePipelineBounds);
        }
        
        
        int projectionFunctorIndex(int level, bool isLeft) {
            int result = 1 + level * NUM_FRAGMENTS_PER_COMPOSITE_TASK + (isLeft ? 0 : 1);
            return result;
        }
        
        
        void RenderSpace::prepareProjectionFunctors() {
            mProjectionFunctors = new vector<CompositeProjectionFunctor*>();
            mProjectionFunctors->push_back(NULL);//skip position zero
            // level 0
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<1>());
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            // level 1
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<2>());
            // level 2
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<4>());
            // level 3
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<8>());
            // level 4
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<16>());
            // level 5
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<32>());
            // level 6
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<64>());
            // level 7
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<128>());
            // level 8
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<256>());
            // level 9
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<512>());
            // level 10
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<0>());
            mProjectionFunctors->push_back(new CompositeProjectionFunctorClass<1024>());
            assert(log2f(mImageSize.depth) <= 10);
            
            for(int i = 0; i < mProjectionFunctors->size() - 1; i += NUM_FRAGMENTS_PER_COMPOSITE_TASK) {
                int level = i / NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                int functorID0 = projectionFunctorIndex(level, true);
                mRuntime->register_projection_functor(functorID0, (*mProjectionFunctors)[functorID0]);
                int functorID1 = projectionFunctorIndex(level, false);
                mRuntime->register_projection_functor(functorID1, (*mProjectionFunctors)[functorID1]);
            }
        }
        
        void RenderSpace::partitionImageForComposite() {
            fragmentImageLayers();
            prepareCompositeLaunchDomains();
            prepareProjectionFunctors();
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
            Point<DIMENSIONS> origin = imageSize.origin();
            origin.x[2] = layer;
            Point<DIMENSIONS> upperBound = imageSize.upperBound() - Point<DIMENSIONS>::ONES();
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
        
        
        inline void RenderSpace::compositeTwoPixelsOver(PixelField *r0,
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
        
        
        inline RenderSpace::CompositeFunction RenderSpace::compositeFunctionPointer(int depthFunction, int blendFunction) {
            CompositeFunction result = NULL;
            if(depthFunction != 0) {
                result = compositeTwoPixelsOver;
            } else if(blendFunction != 0) {
                
            }
            return result;
        }
        
        
        inline PhysicalRegion RenderSpace::compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, PhysicalRegion region1) {
            
            ByteOffset stride[DIMENSIONS];
            PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
            createImageFieldPointers(args.imageSize, region0, args.layer0, r0, g0, b0, a0, z0, userdata0, stride);
            PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
            createImageFieldPointers(args.imageSize, region1, args.layer1, r1, g1, b1, a1, z1, userdata1, stride);
            
            CompositeFunction compositeFunction = compositeFunctionPointer(args.depthFunction, args.blendFunction);
            
            for(int i = 0; i < args.imageSize.numPixelsPerFragment(); ++i) {
                compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0);
                r0++; g0++; b0++; a0++; z0++; userdata0++;
                r1++; g1++; b1++; a1++; z1++; userdata1++;
            }
            return region0;
        }
        
        
        
        
        void RenderSpace::composite_task(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, HighLevelRuntime *runtime) {
            
            UsecTimer composite(describeTask(task) + " leaf:");
            composite.start();
            PhysicalRegion fragment0 = regions[0];
            PhysicalRegion fragment1 = regions[1];
            CompositeArguments args = ((CompositeArguments*)task->args)[0];
            
#if NULL_COMPOSITE_TASKS
            return;//performance testing
#endif
            
            PhysicalRegion compositedResult = compositeTwoFragments(args, fragment0, fragment1);
            composite.stop();
            cout << composite.to_string() << endl;
        }
        
        
        
        void RenderSpace::addArgumentsToLauncher(Legion::IndexTaskLauncher &launcher, int layer0, int layer1, int taskZ, Legion::ArgumentMap &argMap) {
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            Point<DIMENSIONS> point = Point<DIMENSIONS>::ZEROES();
            point.x[2] = taskZ;
            CompositeArguments args = { mImageSize, layer0, layer1, 0, 0 };
            
            for(int fragment = 0; fragment < mImageSize.numFragmentsPerLayer; ++fragment) {
                DomainPoint domainPoint = DomainPoint::from_point<DIMENSIONS>(point);
                argMap.set_point(domainPoint, TaskArgument(&args, sizeof(args)));
                point = mImageSize.incrementFragment(point);
            }
        }
        
        
        void RenderSpace::addTreeRegionRequirementToLauncher(Legion::IndexTaskLauncher &launcher, int level, bool isLeft, PrivilegeMode privilege, CoherenceProperty coherence) {
            int projectionFunctorID = projectionFunctorIndex(level, isLeft);
//            RegionRequirement req(mCompositePartition, projectionFunctorID, privilege, coherence, mImage);
            RegionRequirement req(mCompositePartition, 0, privilege, coherence, mImage);
            addImageFieldsToRequirement(req);
            launcher.add_region_requirement(req);
        }
        
        
        FutureMap RenderSpace::launchTreeLevel(int level, int ordering[]) {
            ArgumentMap argMap;
            IndexTaskLauncher treeLauncher(mCompositeTaskID, mCompositeTreeDomain, TaskArgument(NULL, 0), argMap);
            addTreeRegionRequirementToLauncher(treeLauncher, level, true, READ_WRITE, SIMULTANEOUS);
            addTreeRegionRequirementToLauncher(treeLauncher, level, false, READ_ONLY, SIMULTANEOUS);
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            
            int increment = (int)powf(2.0, (float)level);
            for(int i = 0; i < mImageSize.depth; i += increment * NUM_FRAGMENTS_PER_COMPOSITE_TASK) {
                int taskZ = i / NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                addArgumentsToLauncher(treeLauncher, i, i + increment, taskZ, argMap);
            }
            FutureMap futures = mRuntime->execute_index_space(mContext, treeLauncher);
            return futures;
        }
        
        
        
        FutureMap RenderSpace::launchCompositeTaskTree(int ordering[]) {
            int numTreeLevels = (int)log2f((float)mImageSize.depth);
            FutureMap futures;
            for(int level = 0; level < numTreeLevels; ++level) {
                futures = launchTreeLevel(level, ordering);
                
                futures.wait_all_results();////
                
            }
            return futures;
        }
        
        
        FutureMap RenderSpace::reduceAssociative(int ordering[]) {
            return launchCompositeTaskTree(ordering);
        }
        
        FutureMap RenderSpace::reduceAssociativeCommutative(){
            return reduceAssociative(defaultPermutation());
        }
        
        FutureMap RenderSpace::reduceAssociativeNoncommutative(int ordering[]){
            return reduceAssociative(ordering);
        }
        
        FutureMap RenderSpace::launchCompositeTaskPipeline(int ordering[]) {
            ArgumentMap argMap;
            IndexTaskLauncher pipelineLauncher(mCompositeTaskID, mCompositePipelineDomain, TaskArgument(NULL, 0), argMap);
            
            int projectionFunctorID0 = projectionFunctorIndex(0, true);
            RegionRequirement req0(mCompositePartition, projectionFunctorID0, READ_WRITE, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req0);
            pipelineLauncher.add_region_requirement(req0);
            
            int projectionFunctorID1 = projectionFunctorIndex(0, false);
            RegionRequirement req1(mCompositePartition, projectionFunctorID1, READ_ONLY, SIMULTANEOUS, mImage);
            addImageFieldsToRequirement(req1);
            pipelineLauncher.add_region_requirement(req1);
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            
            for(int i = mImageSize.depth; i > 0; i -= NUM_FRAGMENTS_PER_COMPOSITE_TASK) {
                int layer = i - NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                addArgumentsToLauncher(pipelineLauncher, layer, layer + 1, layer, argMap);
            }
            
            FutureMap futures = mRuntime->execute_index_space(mContext, pipelineLauncher);
            return futures;
        }
        
        FutureMap RenderSpace::reduceNonassociative(int ordering[]) {
            return launchCompositeTaskPipeline(ordering);
        }
        
        FutureMap RenderSpace::reduceNonassociativeCommutative(){
            return reduceNonassociative(defaultPermutation());
        }
        
        FutureMap RenderSpace::reduceNonassociativeNoncommutative(int ordering[]){
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
            TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
            LogicalRegion displayPlane = createDisplayPlane();
            RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
            Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
            return displayFuture;
        }
        
    }
}
