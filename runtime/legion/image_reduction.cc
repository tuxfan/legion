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

#include "image_reduction.h"

#include <iostream>
#include <fstream>
#include <math.h>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
    namespace Visualization {
        
        ImageReduction::ImageReduction(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime) {
            mImageSize = imageSize;
            mContext = ctx;
            mRuntime = runtime;
            mDefaultPermutation = NULL;
            mDepthFunction = 0;
            mBlendFunctionSource = 0;
            mBlendFunctionDestination = 0;
            createImage();
            partitionImageByDepth();
            prepareImageForComposite();
            registerTasks();
        }
        
        ImageReduction::~ImageReduction() {
            if(mDefaultPermutation != NULL) {
                delete [] mDefaultPermutation;
                mDefaultPermutation = NULL;
            }
        }
        
        
        
        void ImageReduction::registerTasks() {
            
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
        
        
        FieldSpace ImageReduction::imageFields() {
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
        
        
        void ImageReduction::createImage() {
            Rect<DIMENSIONS> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<DIMENSIONS>::ONES());
            mImageDomain = Domain::from_rect<DIMENSIONS>(imageBounds);
            //TODO mRuntime->attach_name(mImageDomain, "image domain");
            IndexSpace pixels = mRuntime->create_index_space(mContext, mImageDomain);
            mRuntime->attach_name(pixels, "image index space");
            FieldSpace fields = imageFields();
            mImage = mRuntime->create_logical_region(mContext, pixels, fields);
            mRuntime->attach_name(mImage, "image");
        }
        
        
        void ImageReduction::partitionImageByDepth() {
            Blockify<DIMENSIONS> coloring(mImageSize.layerSize());
            IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageDepthIndexPartition, "image depth index partition");
            mDepthPartition = mRuntime->get_logical_partition(mContext, mImage, imageDepthIndexPartition);
            
            Rect<DIMENSIONS> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<DIMENSIONS>::ONES());
            mDepthDomain = Domain::from_rect<DIMENSIONS>(depthBounds);
            //TODO mRuntime->attach_name(mDepthDomain, "depth domain");
        }
        
        
        void ImageReduction::prepareCompositePartition() {
            Blockify<DIMENSIONS> coloring(mImageSize.fragmentSize());
            IndexPartition imageCompositeIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageCompositeIndexPartition, "image composite index partition");
            mCompositePartition = mRuntime->get_logical_partition(mContext, mImage, imageCompositeIndexPartition);
        }
        
        
        
        Domain ImageReduction::compositeDomain(int increment) {
            Point<DIMENSIONS> numTreeComposites = mImageSize.numFragments();
            numTreeComposites.x[2] /= (NUM_FRAGMENTS_PER_COMPOSITE_TASK * increment);
            Rect<DIMENSIONS> compositeTreeBounds(mImageSize.origin(), numTreeComposites - Point<DIMENSIONS>::ONES());
            return Domain::from_rect<DIMENSIONS>(compositeTreeBounds);
            
        }
        
        
        int ImageReduction::numTreeLevels() {
            int numTreeLevels = log2f(mImageSize.depth);
            if(powf(2.0f, numTreeLevels) < mImageSize.depth) {
                numTreeLevels++;
            }
            return numTreeLevels;
        }
        
        
        void ImageReduction::prepareCompositeDomains() {
            int increment = 1;
            
            mCompositeTreeDomain = vector<Domain>();
            for(int level = 0; level < numTreeLevels(); ++level) {
                mCompositeTreeDomain.push_back(compositeDomain(increment));
            }
            
            Rect<DIMENSIONS> compositePipelineBounds(mImageSize.origin(), mImageSize.numFragments() - Point<DIMENSIONS>::ONES());
            mCompositePipelineDomain = Domain::from_rect<DIMENSIONS>(compositePipelineBounds);
        }
        
        
        
        void ImageReduction::prepareProjectionFunctors() {
            mFunctor0 = new CompositeProjectionFunctor<0>(1);
            mRuntime->register_projection_functor(mFunctor0->functorID(), mFunctor0);
            mFunctor1 = new CompositeProjectionFunctor<1>(2);
            mRuntime->register_projection_functor(mFunctor1->functorID(), mFunctor1);
        }
        
        void ImageReduction::prepareImageForComposite() {
            prepareCompositePartition();
            prepareCompositeDomains();
            prepareProjectionFunctors();
        }
        
        
        void ImageReduction::addImageFieldsToRequirement(RegionRequirement &req) {
            req.add_field(FID_FIELD_R);
            req.add_field(FID_FIELD_G);
            req.add_field(FID_FIELD_B);
            req.add_field(FID_FIELD_A);
            req.add_field(FID_FIELD_Z);
            req.add_field(FID_FIELD_USERDATA);
        }
        
        
        void ImageReduction::createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                     Rect<DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]) {
            acc = region.get_field_accessor(fieldID).typeify<PixelField>();
            Rect<DIMENSIONS> tempBounds;
            field = acc.raw_rect_ptr<DIMENSIONS>(imageBounds, tempBounds, offset);
            assert(imageBounds == tempBounds);
        }
        
        
        void ImageReduction::create_image_field_pointers(ImageSize imageSize,
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
        
        
        FutureMap ImageReduction::launch_task_by_depth(unsigned taskID){
            ArgumentMap argMap;
            IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(&mImageSize, sizeof(ImageSize)), argMap);
            //TODO mRuntime->attach_name(depthLauncher, "depth task launcher");
            RegionRequirement req(mDepthPartition, 0, READ_WRITE, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            depthLauncher.add_region_requirement(req);
            FutureMap futures = mRuntime->execute_index_space(mContext, depthLauncher);
            return futures;
        }
        
        
        
        int *ImageReduction::defaultPermutation(){
            if(mDefaultPermutation == NULL) {
                mDefaultPermutation = new int[mImageSize.depth];
                for(int i = 0; i < mImageSize.depth; ++i) {
                    mDefaultPermutation[i] = i;
                }
            }
            return mDefaultPermutation;
        }
        
        
        inline PhysicalRegion ImageReduction::compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, PhysicalRegion region1) {
            
            ByteOffset stride[DIMENSIONS];
            PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
            PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
            ImageReductionComposite::CompositeFunction* compositeFunction;
            
            create_image_field_pointers(args.imageSize, region0, args.layer0, r0, g0, b0, a0, z0, userdata0, stride);
            create_image_field_pointers(args.imageSize, region1, args.layer1, r1, g1, b1, a1, z1, userdata1, stride);
            compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination);
            compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageSize.numPixelsPerFragment());
            
            return region0;
        }
        
        
        
        
        void ImageReduction::composite_task(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context ctx, HighLevelRuntime *runtime) {
            UsecTimer composite(describe_task(task) + " leaf:");
            composite.start();
            CompositeArguments args = ((CompositeArguments*)task->local_args)[0];
            if(args.layer1 >= 0) {
                PhysicalRegion fragment0 = regions[0];
                PhysicalRegion fragment1 = regions[1];
                
#if NULL_COMPOSITE_TASKS
                return;//performance testing
#endif
                
                PhysicalRegion compositedResult = compositeTwoFragments(args, fragment0, fragment1);
            }
            composite.stop();
            cout << composite.to_string() << endl;
        }
        
        
        
        void ImageReduction::addCompositeArgumentsToArgmap(CompositeArguments *args, int taskZ, Legion::ArgumentMap &argMap) {
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            Point<DIMENSIONS> point = Point<DIMENSIONS>::ZEROES();
            point.x[2] = taskZ;
            
            for(int fragment = 0; fragment < mImageSize.numFragmentsPerLayer; ++fragment) {
                DomainPoint domainPoint = DomainPoint::from_point<DIMENSIONS>(point);
                argMap.set_point(domainPoint, TaskArgument(args, sizeof(*args)));
                point = mImageSize.incrementFragment(point);
            }
        }
        
        
        void ImageReduction::addRegionRequirementToCompositeLauncher(Legion::IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence) {
            RegionRequirement req(mCompositePartition, projectionFunctorID, privilege, coherence, mImage);
            addImageFieldsToRequirement(req);
            launcher.add_region_requirement(req);
        }
        
        
        FutureMap ImageReduction::launchTreeLevel(int level, int ordering[]) {
            ArgumentMap argMap;
            int increment = powf(2.0f, level);
            int numLayers = mImageSize.depth / (increment * NUM_FRAGMENTS_PER_COMPOSITE_TASK);
            CompositeArguments args[numLayers];
            
            for(int i = 0; i < numLayers; i++) {
                int taskZ = i;
                int layer0 = i * NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                int layer1 = layer0 + increment;
                layer1 = (layer1 < mImageSize.depth) ? layer1 : -1;
                //ordering[layer0], ordering[layer1]
                args[taskZ] = (CompositeArguments){ mImageSize, layer0, layer1, mDepthFunction, mBlendFunctionSource, mBlendFunctionDestination };
                addCompositeArgumentsToArgmap(args + taskZ, taskZ, argMap);
            }
            
            Domain launchDomain = mCompositeTreeDomain[level];
            IndexTaskLauncher treeLauncher(mCompositeTaskID, launchDomain, TaskArgument(NULL, 0), argMap);
            addRegionRequirementToCompositeLauncher(treeLauncher, mFunctor0->functorID(), READ_WRITE, EXCLUSIVE);
            addRegionRequirementToCompositeLauncher(treeLauncher, mFunctor1->functorID(), READ_ONLY, SIMULTANEOUS);
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            
            FutureMap futures = mRuntime->execute_index_space(mContext, treeLauncher);
            return futures;
        }
        
        
        
        FutureMap ImageReduction::launchCompositeTaskTree(int ordering[]) {
            FutureMap futures;
            for(int level = 0; level < numTreeLevels(); ++level) {
                futures = launchTreeLevel(level, ordering);
            }
            return futures;
        }
        
        
        FutureMap ImageReduction::reduceAssociative(int ordering[]) {
            return launchCompositeTaskTree(ordering);
        }
        
        FutureMap ImageReduction::reduce_associative_commutative(){
            return reduceAssociative(defaultPermutation());
        }
        
        FutureMap ImageReduction::reduce_associative_noncommutative(int ordering[]){
            return reduceAssociative(ordering);
        }
        
        FutureMap ImageReduction::launchCompositeTaskPipeline(int ordering[]) {
            ArgumentMap argMap;
            int taskDepth = mImageSize.depth / NUM_FRAGMENTS_PER_COMPOSITE_TASK;
            CompositeArguments args[taskDepth];
            
            for(int i = mImageSize.depth; i > 1; i--) {
                assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
                int layer0 = i - NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                int layer1 = layer0 + 1;
                int taskZ = layer0;
                //ordering[layer0], ordering[layer1]
                args[taskZ] = (CompositeArguments){ mImageSize, layer0, layer1, GL_LESS, 0, 0 };
                addCompositeArgumentsToArgmap(args + taskZ, taskZ, argMap);
            }
            
            IndexTaskLauncher pipelineLauncher(mCompositeTaskID, mCompositePipelineDomain, TaskArgument(NULL, 0), argMap);
            addRegionRequirementToCompositeLauncher(pipelineLauncher, mFunctor0->functorID(), READ_WRITE, EXCLUSIVE);
            addRegionRequirementToCompositeLauncher(pipelineLauncher, mFunctor1->functorID(), READ_ONLY, SIMULTANEOUS);
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            
            FutureMap futures = mRuntime->execute_index_space(mContext, pipelineLauncher);
            return futures;
        }
        
        FutureMap ImageReduction::reduceNonassociative(int ordering[]) {
            return launchCompositeTaskPipeline(ordering);
        }
        
        FutureMap ImageReduction::reduce_nonassociative_commutative(){
            return reduceNonassociative(defaultPermutation());
        }
        
        FutureMap ImageReduction::reduce_nonassociative_noncommutative(int ordering[]){
            return reduceNonassociative(ordering);
        }
        
        
        
        void ImageReduction::display_task(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context ctx, HighLevelRuntime *runtime) {
            
            DisplayArguments args = ((DisplayArguments*)task->args)[0];
            char fileName[1024];
            sprintf(fileName, "display.%d.txt", args.t);
            string outputFileName = string(fileName);
            UsecTimer display(describe_task(task) + " write to " + outputFileName + ":");
            display.start();
            PhysicalRegion displayPlane = regions[0];
            ByteOffset stride[DIMENSIONS];
            PixelField *r, *g, *b, *a, *z, *userdata;
            create_image_field_pointers(args.imageSize, displayPlane, args.imageSize.depth - 1, r, g, b, a, z, userdata, stride);
            
            FILE *outputFile = fopen(outputFileName.c_str(), "wb");
            fwrite(r, 6 * sizeof(*r), args.imageSize.pixelsPerLayer(), outputFile);
            fclose(outputFile);
            
            display.stop();
            cout << display.to_string() << endl;
        }
        
        
        
        Future ImageReduction::display(int t) {
            DisplayArguments args = { mImageSize, t };
            TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
            DomainPoint origin = DomainPoint::from_point<DIMENSIONS>(Point<DIMENSIONS>::ZEROES());
            LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mDepthPartition, origin);
            RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
            Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
            return displayFuture;
        }
        
    }
}
