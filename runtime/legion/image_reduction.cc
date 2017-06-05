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
      partitionImageByScreenSpace();
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
      
      mScreenSpaceTaskID = mRuntime->generate_dynamic_task_id();
      TaskVariantRegistrar screenSpaceRegistrar(mScreenSpaceTaskID, "screenSpaceTask");
      screenSpaceRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(1/*index*/, layoutConstraintID);
      mRuntime->register_task_variant<screen_space_task>(screenSpaceRegistrar);
      mRuntime->attach_name(mScreenSpaceTaskID, "screenSpaceTask");
      
      mCompositeTaskID = mRuntime->generate_dynamic_task_id();
      TaskVariantRegistrar compositeRegistrar(mCompositeTaskID, "compositeTask");
      compositeRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, layoutConstraintID);
      mRuntime->register_task_variant<int, composite_task>(compositeRegistrar);
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
      IndexSpace pixels = mRuntime->create_index_space(mContext, mImageDomain);
      mRuntime->attach_name(pixels, "image index space");
      FieldSpace fields = imageFields();
      mImage = mRuntime->create_logical_region(mContext, pixels, fields);
      mRuntime->attach_name(mImage, "image");
    }
    
    
    void ImageReduction::partitionImageByDepth() {
      Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(mImageSize.layerSize());
      IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
      mDepthPartition = mRuntime->get_logical_partition(mContext, mImage, imageDepthIndexPartition);
      mRuntime->attach_name(mDepthPartition, "image depth partition");
      Rect<IMAGE_REDUCTION_DIMENSIONS> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
      mDepthDomain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(depthBounds);
    }
    
    
    void ImageReduction::partitionImageByScreenSpace() {
      Point<IMAGE_REDUCTION_DIMENSIONS> partitionSize = mImageSize.fragmentSize();
      partitionSize.x[2] = mImageSize.depth;
      Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(partitionSize);
      IndexPartition imageCompositeScreenSpaceIndexPartition = mRuntime->create_index_partition(mContext, mImage.get_index_space(), coloring);
      mScreenSpacePartition = mRuntime->get_logical_partition(mContext, mImage, imageCompositeScreenSpaceIndexPartition);
      mRuntime->attach_name(mScreenSpacePartition, "image screen space partition");
      Point<IMAGE_REDUCTION_DIMENSIONS> screenSpaceBounds = mImageSize.numFragments();
      screenSpaceBounds.x[2] = 1;
      Rect<IMAGE_REDUCTION_DIMENSIONS> screenSpaceRect(mImageSize.origin(), screenSpaceBounds - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
      mScreenSpaceDomain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(screenSpaceRect);
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
    
    
    FutureMap ImageReduction::launch_task_by_depth(unsigned taskID, void *args, int argLen){
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageSize) + argLen;
      char argsBuffer[totalArgLen];
      memcpy(argsBuffer, &mImageSize, sizeof(mImageSize));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageSize), args, argLen);
      }
      
      IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(argsBuffer, totalArgLen), argMap);
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
    
    
    
    
    int ImageReduction::composite_task(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, HighLevelRuntime *runtime) {
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
      return (args.layer0);//output destination
    }
    
    
    
    
    void ImageReduction::addSubregionRequirementToFragmentLauncher(TaskLauncher &launcher, DomainPoint origin, int layer,
                                                                   Context context, Runtime* runtime, LogicalPartition partition, LogicalRegion parent) {
      origin[2] = layer;
      LogicalRegion subregion = runtime->get_logical_subregion_by_color(context, partition, origin);
      RegionRequirement req(subregion, READ_WRITE, EXCLUSIVE, parent);
      addImageFieldsToRequirement(req);
      launcher.add_region_requirement(req);
    }
    
    
    Future ImageReduction::launchCompositeTask(DomainPoint origin, int taskNumber, ScreenSpaceArguments args, LogicalRegion parent,
                                               FutureSet futures, int layer0, int layer1,
                                               Context context, Runtime* runtime, LogicalPartition fragmentPartition) {
      CompositeArguments compositeArgs = { args.imageSize, origin[0], origin[1], layer0, layer1,
        args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination };
      TaskLauncher compositeLauncher(args.compositeTaskID, TaskArgument(&compositeArgs, sizeof(compositeArgs)));
      addSubregionRequirementToFragmentLauncher(compositeLauncher, origin, layer0, context, runtime, fragmentPartition, parent);
      addSubregionRequirementToFragmentLauncher(compositeLauncher, origin, layer1, context, runtime, fragmentPartition, parent);
      Future future = runtime->execute_task(context, compositeLauncher);
      return future;
    }
    
    
    ImageReduction::FutureSet ImageReduction::launchTreeReduction(ScreenSpaceArguments args, int *ordering, LogicalRegion parent, int treeLevel,
                                                                  int maxTreeLevel, DomainPoint origin,
                                                                  Context context, Runtime* runtime, LogicalPartition fragmentPartition) {
      
      FutureSet futures;
      if(treeLevel < maxTreeLevel - 1) {
        futures = launchTreeReduction(args, ordering, parent, treeLevel + 1, maxTreeLevel, origin, context, runtime, fragmentPartition);
      }
      
      int numTasks = (int)(powf(2.0f, treeLevel));
      FutureSet result = FutureSet();
      for(int i = 0; i < numTasks; ++i) {
        int layer0;
        int layer1;
        if(treeLevel < maxTreeLevel - 1) {
          layer0 = futures[i * NUM_FRAGMENTS_PER_COMPOSITE_TASK].get_result<int>();
          layer1 = futures[i * NUM_FRAGMENTS_PER_COMPOSITE_TASK + 1].get_result<int>();
        } else {
          layer0 = ordering[i * NUM_FRAGMENTS_PER_COMPOSITE_TASK];
          layer1 = ordering[i * NUM_FRAGMENTS_PER_COMPOSITE_TASK + 1];
        }
        Future future = launchCompositeTask(origin, i, args, parent, futures, layer0, layer1, context, runtime, fragmentPartition);
        result.push_back(future);
      }
      return result;
    }
    
    
    void ImageReduction::launchPipelineReduction(ScreenSpaceArguments args, int *ordering) {
      
    }
    
    
    int ImageReduction::numTreeLevels(ImageSize imageSize) {
      int numTreeLevels = log2f(imageSize.depth);
      if(powf(2.0f, numTreeLevels) < imageSize.depth) {
        numTreeLevels++;
      }
      return numTreeLevels;
    }
    
    int ImageReduction::subtreeHeight(ImageSize imageSize) {
      const int totalLevels = numTreeLevels(imageSize);
      const int MAX_LEVELS_PER_SUBTREE = 7; // 128 tasks per subtree
      return (totalLevels < MAX_LEVELS_PER_SUBTREE) ? totalLevels : MAX_LEVELS_PER_SUBTREE;
    }
    
    
    static int colorIndex(DomainPoint p, ImageSize imageSize) {
      Point<IMAGE_REDUCTION_DIMENSIONS> size = imageSize.numFragments();
      int result = 0;
      for(int i = 0; i < IMAGE_REDUCTION_DIMENSIONS - 1; ++i) {
        result *= (p[i] * size.x[i]);
      }
      return result;
    }
    
    
    
    LogicalPartition ImageReduction::partitionScreenSpaceByFragment(DomainPoint origin, Runtime* runtime, Context context, LogicalRegion parent, ImageSize imageSize) {
      
      Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(imageSize.fragmentSize());
      IndexPartition imageFragmentIndexPartition = runtime->create_index_partition(context, parent.get_index_space(), coloring);
      LogicalPartition fragmentPartition = runtime->get_logical_partition(context, parent, imageFragmentIndexPartition);
      runtime->attach_name(fragmentPartition, "fragment partition");
      return fragmentPartition;
    }
    
    
    
    void ImageReduction::screen_space_task(const Task *task,
                                           const std::vector<PhysicalRegion> &regions,
                                           Context ctx, HighLevelRuntime *runtime) {
      ScreenSpaceArguments args = ((ScreenSpaceArguments*)task->args)[0];
      int* ordering = (int*)((char*)task->args + sizeof(args));
      LogicalRegion subregion = task->regions[0].region;
      
      LogicalPartition fragmentPartition = partitionScreenSpaceByFragment(task->index_point, runtime, ctx, subregion, args.imageSize);
      
      if(args.isAssociative) {
        launchTreeReduction(args, ordering, subregion, 0, subtreeHeight(args.imageSize), task->index_point, ctx, runtime, fragmentPartition);
      } else {
        launchPipelineReduction(args, ordering);
      }
      
    }
    
    
    
    char* ImageReduction::screenSpaceArguments(int ordering[], bool isAssociative) {
      int orderingSize = sizeof(ordering[0]) * mImageSize.depth;
      int totalSize = sizeof(ScreenSpaceArguments) + orderingSize;
      ScreenSpaceArguments args = { mImageSize, totalSize, isAssociative, mCompositeTaskID,
        mDepthFunction, mBlendFunctionSource, mBlendFunctionDestination, mImage };
      char *result = new char[totalSize];
      memcpy(result, &args, sizeof(args));
      memcpy(result + sizeof(args), ordering, orderingSize);
      return result;
    }
    
    
    FutureMap ImageReduction::launchScreenSpaceTasks(int ordering[], bool isAssociative) {
      ArgumentMap argMap;
      char* args = screenSpaceArguments(ordering, isAssociative);
      ScreenSpaceArguments* screenSpaceArgs = (ScreenSpaceArguments*)args;
      IndexTaskLauncher screenSpaceLauncher(mScreenSpaceTaskID, mScreenSpaceDomain, TaskArgument(args, screenSpaceArgs->totalSize), argMap);
      RegionRequirement req(mScreenSpacePartition, 0, READ_WRITE, EXCLUSIVE, mImage);
      addImageFieldsToRequirement(req);
      screenSpaceLauncher.add_region_requirement(req);
      
      FutureMap futures = mRuntime->execute_index_space(mContext, screenSpaceLauncher);
      delete [] args;
      return futures;
    }
    
    
    FutureMap ImageReduction::reduceAssociative(int ordering[]) {
      return launchScreenSpaceTasks(ordering, true);
    }
    
    FutureMap ImageReduction::reduce_associative_commutative(){
      return reduceAssociative(defaultPermutation());
    }
    
    FutureMap ImageReduction::reduce_associative_noncommutative(int ordering[]){
      return reduceAssociative(ordering);
    }
    
    
    FutureMap ImageReduction::reduceNonassociative(int ordering[]) {
      return launchScreenSpaceTasks(ordering, false);
    }
    
    FutureMap ImageReduction::reduce_nonassociative_commutative(){
      return reduceNonassociative(defaultPermutation());
    }
    
    FutureMap ImageReduction::reduce_nonassociative_noncommutative(int ordering[]){
      return reduceNonassociative(ordering);
    }
    
    //this matrix project pixels from world space bounded by l,r,t,p,f,n to screen space -1,1
    // for an index task launch we mwant an ffine map from a domain launch index to a subregions index in the partition
    
    static float *homogeneousOrthographicProjectionMatrix(float right, float left, float top, float bottom, float near, float far) {
      static float *result = NULL;
      if(result == NULL) {
        result = new float[16];
        result[0] = (2.0f / (right - left));
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = -((right + left) / (right - left));
        result[4] = 0.0f;
        result[5] = (2.0f / (top - bottom));
        result[6] = 0.0f;
        result[7] = -((top + bottom) / (top - bottom));
        result[8] = 0.0f;
        result[9] = 0.0f;
        result[10] = (-2.0f / (far - near));
        result[11] = -((far + near) / (far - near));
        result[12] = 0.0f;
        result[13] = 0.0f;
        result[14] = 0.0f;
        result[15] = 1.0f;
      }
      return result;
    }
    
    static inline int index(int i, int j) {
      return i * 4 + j;
    }
    
    
    // this returns the composite order for a simulation subdomain
    static int subdomainToCompositeIndex(float l0, float r0, float t0, float b0, float f0, float n0,
                                         float left, float right, float top, float bottom, float far, float near, int scale) {
      float centroid[] = { (r0 - l0) / 2.0f, (t0 - b0) / 2.0f, (f0 - n0) / 2.0f, 1.0f };
      float projection[] = { 0.0f, 0.0f, 0.0f, 0.0f };
      const int numHomgeneousCoordinates = 4;
      float *matrix = homogeneousOrthographicProjectionMatrix(right, left, top, bottom, near, far);
      for(int i = 0; i < numHomgeneousCoordinates; ++i) {
        for(int j = 0; j < numHomgeneousCoordinates; ++j) {
          projection[i] += centroid[i] * matrix[index(i, j)];
        }
      }
      float z = projection[1] / projection[3];//range -1 to 1
      int scaledZ = (z + 1.0f) * (scale / 2);
      return scaledZ;
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
