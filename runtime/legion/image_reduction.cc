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
        
#ifndef DYNAMIC_PROJECTION_FUNCTOR_REGISTRATION_WORKS_ON_MULTIPLE_NODES
        ImageReduction::CompositeProjectionFunctor<0>* ImageReduction::mFunctor0;
        ImageReduction::CompositeProjectionFunctor<1>* ImageReduction::mFunctor1;
#endif
        
        ImageReduction::ImageReduction(ImageSize imageSize, Context context, HighLevelRuntime *runtime) {
            mImageSize = imageSize;
            mContext = context;
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
            .add_layout_constraint_set(1/*index*/, layoutConstraintID);
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
            Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
            mImageDomain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(imageBounds);
            //TODO mRuntime->attach_name(mImageDomain, "image domain");
            IndexSpace pixels = mRuntime->create_index_space(mContext, mImageDomain);
            mRuntime->attach_name(pixels, "image index space");
            FieldSpace fields = imageFields();
            mImage = mRuntime->create_logical_region(mContext, pixels, fields);
            mRuntime->attach_name(mImage, "image");
        }
        
        
        void ImageReduction::partitionImageByDepth() {
            Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(mImageSize.layerSize());
            IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageDepthIndexPartition, "image depth index partition");
            mDepthPartition = mRuntime->get_logical_partition(mContext, mImage, imageDepthIndexPartition);
            
            Rect<IMAGE_REDUCTION_DIMENSIONS> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
            mDepthDomain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(depthBounds);
            //TODO mRuntime->attach_name(mDepthDomain, "depth domain");
        }
        
        
        void ImageReduction::prepareCompositePartition() {
            Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(mImageSize.fragmentSize());
            IndexPartition imageCompositeIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
            mRuntime->attach_name(imageCompositeIndexPartition, "image composite index partition");
            mCompositePartition = mRuntime->get_logical_partition(mContext, mImage, imageCompositeIndexPartition);
        }
        
        
        
        Domain ImageReduction::compositeDomain(int increment) {
            Point<IMAGE_REDUCTION_DIMENSIONS> numTreeComposites = mImageSize.numFragments();
            numTreeComposites.x[2] /= (NUM_FRAGMENTS_PER_COMPOSITE_TASK * increment);
            Rect<IMAGE_REDUCTION_DIMENSIONS> compositeTreeBounds(mImageSize.origin(), numTreeComposites - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
            return Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(compositeTreeBounds);
            
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
                increment *= 2;
            }
            
            Rect<IMAGE_REDUCTION_DIMENSIONS> compositePipelineBounds(mImageSize.origin(), mImageSize.numFragments() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
            mCompositePipelineDomain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(compositePipelineBounds);
        }
        
        
        
        void ImageReduction::prepareProjectionFunctors() {
            mFunctor0 = new CompositeProjectionFunctor<0>(1);
#ifdef DYNAMIC_PROJECTION_FUNCTOR_REGISTRATION_WORKS_ON_MULTIPLE_NODES
            mRuntime->register_projection_functor(mFunctor0->functorID(), mFunctor0);
#else
            Legion::Runtime::preregister_projection_functor(mFunctor0->functorID(), mFunctor0);
#endif
            mFunctor1 = new CompositeProjectionFunctor<1>(2);
#ifdef DYNAMIC_PROJECTION_FUNCTOR_REGISTRATION_WORKS_ON_MULTIPLE_NODES
            mRuntime->register_projection_functor(mFunctor1->functorID(), mFunctor1);
#else
            Legion::Runtime::preregister_projection_functor(mFunctor1->functorID(), mFunctor1);
#endif
        }
        
        void ImageReduction::prepareImageForComposite() {
            prepareCompositePartition();
            prepareCompositeDomains();
#ifdef DYNAMIC_PROJECTION_FUNCTOR_REGISTRATION_WORKS_ON_MULTIPLE_NODES
            prepareProjectionFunctors();
#endif
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
                                                     Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]) {
            acc = region.get_field_accessor(fieldID).typeify<PixelField>();
            Rect<IMAGE_REDUCTION_DIMENSIONS> tempBounds;
            field = acc.raw_rect_ptr<IMAGE_REDUCTION_DIMENSIONS>(imageBounds, tempBounds, offset);
            assert(imageBounds == tempBounds);
        }
        
        
        void ImageReduction::create_image_field_pointers(ImageSize imageSize,
                                                         PhysicalRegion region,
                                                         Point<IMAGE_REDUCTION_DIMENSIONS> origin,
                                                         PixelField *&r,
                                                         PixelField *&g,
                                                         PixelField *&b,
                                                         PixelField *&a,
                                                         PixelField *&z,
                                                         PixelField *&userdata,
                                                         ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS]) {
            
            Rect<IMAGE_REDUCTION_DIMENSIONS> tempBounds;
            Point<IMAGE_REDUCTION_DIMENSIONS> upperBound = origin + imageSize.fragmentSize() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES();
            Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds = Rect<IMAGE_REDUCTION_DIMENSIONS>(origin, upperBound);
            
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
        
        
        inline PhysicalRegion ImageReduction::compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, Point<IMAGE_REDUCTION_DIMENSIONS> origin0,
                                                                    PhysicalRegion region1, Point<IMAGE_REDUCTION_DIMENSIONS> origin1) {
            
            ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
            PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
            PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
            ImageReductionComposite::CompositeFunction* compositeFunction;
            
            create_image_field_pointers(args.imageSize, region0, origin0, r0, g0, b0, a0, z0, userdata0, stride);
            create_image_field_pointers(args.imageSize, region1, origin1, r1, g1, b1, a1, z1, userdata1, stride);
            compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination);
            compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageSize.numPixelsPerFragment());
            
            return region0;
        }
        
        
        
        
        void ImageReduction::composite_task(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context ctx, HighLevelRuntime *runtime) {
            //            UsecTimer composite(describe_task(task) + " leaf:");
            //            composite.start();
            CompositeArguments args = ((CompositeArguments*)task->local_args)[0];
            if(args.layer1 >= 0) {
                PhysicalRegion fragment0 = regions[0];
                PhysicalRegion fragment1 = regions[1];
                Point<IMAGE_REDUCTION_DIMENSIONS> origin0;
                origin0.x[0] = args.x;
                origin0.x[1] = args.y;
                origin0.x[2] = args.layer0;
                Point<IMAGE_REDUCTION_DIMENSIONS> origin1;
                origin1.x[0] = args.x;
                origin1.x[1] = args.y;
                origin1.x[2] = args.layer1;
                
                
#if NULL_COMPOSITE_TASKS
                return;//performance testing
#endif
                
                PhysicalRegion compositedResult = compositeTwoFragments(args, fragment0, origin0, fragment1, origin1);
            }
            //            composite.stop();
            //            cout << composite.to_string() << endl;
        }
        
        
        
        void ImageReduction::addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, Legion::ArgumentMap &argMap, int layer0, int layer1) {
            assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
            Point<IMAGE_REDUCTION_DIMENSIONS> point = Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES();
            point.x[2] = taskZ;
            
            for(int fragment = 0; fragment < mImageSize.numFragmentsPerLayer; ++fragment) {
                DomainPoint domainPoint = DomainPoint::from_point<IMAGE_REDUCTION_DIMENSIONS>(point);
                *argsPtr = (CompositeArguments){ mImageSize, point.x[0], point.x[1], layer0, layer1, mDepthFunction, mBlendFunctionSource, mBlendFunctionDestination };
                argMap.set_point(domainPoint, TaskArgument(argsPtr, sizeof(CompositeArguments)));
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
            int numTasks = mImageSize.depth / (increment * NUM_FRAGMENTS_PER_COMPOSITE_TASK);
            CompositeArguments args[numTasks * mImageSize.numFragmentsPerLayer];
            CompositeArguments *argsPtr = args;
            
            for(int i = 0; i < numTasks; i++) {
                int taskZ = i;
                int order0 = i * NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                int layer0 = ordering[order0];
                int order1 = order0 + increment;
                int layer1 = (order1 < mImageSize.depth) ? ordering[order1] : -1;
                addCompositeArgumentsToArgmap(argsPtr, taskZ, argMap, layer0, layer1);
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
            CompositeArguments args[taskDepth * mImageSize.numFragmentsPerLayer];
            CompositeArguments *argsPtr = args;
            
            for(int i = mImageSize.depth; i > 1; i--) {
                assert(NUM_FRAGMENTS_PER_COMPOSITE_TASK == 2);
                int layer0 = i - NUM_FRAGMENTS_PER_COMPOSITE_TASK;
                int layer1 = layer0 + 1;
                int taskZ = layer0;
                addCompositeArgumentsToArgmap(argsPtr, taskZ, argMap, layer0, layer1);
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
            ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
            PixelField *r, *g, *b, *a, *z, *userdata;
            Point<IMAGE_REDUCTION_DIMENSIONS> origin = Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES();
            create_image_field_pointers(args.imageSize, displayPlane, origin, r, g, b, a, z, userdata, stride);
            
            FILE *outputFile = fopen(outputFileName.c_str(), "wb");
            fwrite(r, 6 * sizeof(*r), args.imageSize.pixelsPerLayer(), outputFile);
            fclose(outputFile);
            
            display.stop();
            cout << display.to_string() << endl;
        }
        
        
        
        Future ImageReduction::display(int t) {
            DisplayArguments args = { mImageSize, t };
            TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
            DomainPoint origin = DomainPoint::from_point<IMAGE_REDUCTION_DIMENSIONS>(Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES());
            LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mDepthPartition, origin);
            RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mImage);
            addImageFieldsToRequirement(req);
            taskLauncher.add_region_requirement(req);
            Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
            return displayFuture;
        }
        
    }
}
