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
      mDepthFunction = 0;
      mBlendFunctionSource = 0;
      mBlendFunctionDestination = 0;
      mMySimulationBounds = NULL;
      mNodeID = NULL;
      mAccessorProjectionFunctor = NULL;
      
      createImage(mSourceImage, mSourceImageDomain);
      partitionImageByDepth(mSourceImage, mDepthDomain, mDepthPartition);
      partitionImageByFragment(mSourceImage, mSourceFragmentDomain, mSourceFragmentPartition);
      
      registerTasks();
      initializeNodes();
      assert(mNodeCount > 0);
      mLocalCopyOfNodeID = mNodeID[mNodeCount - 1];//written by initial_task
      initializeViewMatrix();
      createTreeDomains(mLocalCopyOfNodeID, numTreeLevels(imageSize), runtime, imageSize);
    }
    
    ImageReduction::~ImageReduction() {
      if(mMySimulationBounds != NULL) {
        delete [] mMySimulationBounds;
      }
    }
    
    
    int *ImageReduction::mNodeID;
    ImageReduction::SimulationBoundsCoordinate *ImageReduction::mMySimulationBounds;
    ImageReduction::SimulationBoundsCoordinate *ImageReduction::mSimulationBounds;
    int ImageReduction::mNumSimulationBounds;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMin;
    ImageReduction::AccessorProjectionFunctor **ImageReduction::mAccessorProjectionFunctor;
    std::vector<ImageReduction::CompositeProjectionFunctor*> ImageReduction::mCompositeProjectionFunctor;
    int ImageReduction::mNodeCount;
    std::vector<Domain> ImageReduction::mHierarchicalTreeDomain;
    GLfloat ImageReduction::mViewMatrix[16];
    
    
    // this function should be called prior to starting the Legion runtime
    // its purpose is to copy the domain bounds to all nodes
    
    void ImageReduction::preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) {
      mNodeCount = 0;
      mNumSimulationBounds = numBounds;
      
      int totalElements = numBounds * fieldsPerSimulationBounds;
      mSimulationBounds = new SimulationBoundsCoordinate[totalElements];
      memcpy(mSimulationBounds, bounds, sizeof(SimulationBoundsCoordinate) * totalElements);
      mXMin = mYMin = mZMin = 1.0e+32;
      mXMax = mYMax = mZMax = -mXMin;
      SimulationBoundsCoordinate *bound = mSimulationBounds;
      for(int i = 0; i < numBounds; ++i) {
        mXMin = (bound[0] < mXMin) ? bound[0] : mXMin;
        mYMin = (bound[1] < mYMin) ? bound[1] : mYMin;
        mZMin = (bound[2] < mZMin) ? bound[2] : mZMin;
        mXMax = (bound[3] > mXMax) ? bound[3] : mXMax;
        mYMax = (bound[4] > mYMax) ? bound[4] : mYMax;
        mZMax = (bound[5] > mZMax) ? bound[5] : mZMax;
        bound += fieldsPerSimulationBounds;
      }
      std::cout << "loaded " << numBounds << " simulation subdomains, overall bounds ("
      << mXMin << "," << mXMax << " x " << mYMin << "," << mYMax << " x " << mZMin << "," << mZMax << ")" << std::endl;
      
    }
    
    
    
    void ImageReduction::registerTasks() {
      
      LayoutConstraintRegistrar layoutRegistrar(imageFields(), "layout");
      LayoutConstraintID layoutConstraintID = mRuntime->register_layout(layoutRegistrar);
      
      mInitialTaskID = mRuntime->generate_dynamic_task_id();
      TaskVariantRegistrar initialRegistrar(mInitialTaskID, "initialTask");
      initialRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(1/*index*/, layoutConstraintID);
      mRuntime->register_task_variant<initial_task>(initialRegistrar);
      mRuntime->attach_name(mInitialTaskID, "initialTask");
      
      mAccessorTaskID = mRuntime->generate_dynamic_task_id();
      TaskVariantRegistrar accessorRegistrar(mAccessorTaskID, "accessorTask");
      accessorRegistrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(1/*index*/, layoutConstraintID);
      mRuntime->register_task_variant<accessor_task>(accessorRegistrar);
      mRuntime->attach_name(mAccessorTaskID, "accessorTask");
      
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
    
    
    void ImageReduction::createImage(LogicalRegion &region, Domain &domain) {
      Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
      domain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(imageBounds);
      IndexSpace pixels = mRuntime->create_index_space(mContext, domain);
      mRuntime->attach_name(pixels, "image index space");
      FieldSpace fields = imageFields();
      region = mRuntime->create_logical_region(mContext, pixels, fields);
      mRuntime->attach_name(region, "image");
    }
    
    
    void ImageReduction::partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(mImageSize.layerSize());
      IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, image.get_index_space(), coloring);
      partition = mRuntime->get_logical_partition(mContext, image, imageDepthIndexPartition);
      
      std::cout << "created depth partition " << partition << std::endl;
      
      mRuntime->attach_name(partition, "image depth partition");
      Rect<IMAGE_REDUCTION_DIMENSIONS> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
      domain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(depthBounds);
    }
    
    
    void ImageReduction::partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      Blockify<IMAGE_REDUCTION_DIMENSIONS> coloring(mImageSize.fragmentSize());
      IndexPartition imageFragmentIndexPartition = mRuntime->create_index_partition(mContext, image.get_index_space(), coloring);
      mRuntime->attach_name(imageFragmentIndexPartition, "image fragment index");
      partition = mRuntime->get_logical_partition(mContext, image, imageFragmentIndexPartition);
      mRuntime->attach_name(partition, "image fragment partition");
      Rect<IMAGE_REDUCTION_DIMENSIONS> fragmentBounds(mImageSize.origin(), mImageSize.numFragments() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES());
      domain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(fragmentBounds);
    }
    
    /// these store* and retrieve* methods are due to simulation issues on single or multi nodes
    /// on multiple nodes the static data does not overwrite itself
    /// on single nodes the static data must have as many instances as there are nodes
    /// TODO get rid of this
    
    void ImageReduction::storeMySimulationBounds(SimulationBoundsCoordinate* values, int nodeID, int numNodes) {
#ifdef RUNNING_MULTINODE
      nodeID = 0;
      numNodes = 1;
#endif
      if(mMySimulationBounds == NULL) {
        int numElements = numNodes * fieldsPerSimulationBounds;
        mMySimulationBounds = new SimulationBoundsCoordinate[numElements];
      }
      SimulationBoundsCoordinate* myPtr = retrieveMySimulationBounds(nodeID);
      memcpy(myPtr, values, fieldsPerSimulationBounds * sizeof(SimulationBoundsCoordinate));
      
      std::cout << "node id " << nodeID << " stored my bounds " << myPtr[0] << ","
      << myPtr[1] << "," << myPtr[2] << "," << myPtr[3] << "," << myPtr[4] << "," << myPtr[5] << std::endl;
    }
    
    
    ImageReduction::SimulationBoundsCoordinate* ImageReduction::retrieveMySimulationBounds(int nodeID) {
#ifdef RUNNING_MULTINODE
      nodeID = 0;
#endif
      ImageReduction::SimulationBoundsCoordinate* bounds = mMySimulationBounds + nodeID * fieldsPerSimulationBounds;
      return bounds;
    }
    
    
    void ImageReduction::storeMyNodeID(int nodeID, int numNodes) {
#ifdef RUNNING_MULTINODE
      nodeID = 0;
      numNodes = 1;
#endif
      if(mNodeID == NULL) {
        mNodeID = new int[numNodes];
      }
      mNodeID[nodeID] = nodeID;
      mNodeCount++;
    }
    
    
    
    void ImageReduction::storeMyAccessorProjectionFunctor(int functorID, AccessorProjectionFunctor *functor, int numNodes, int nodeID) {
#ifdef RUNNING_MULTINODE
      nodeID = 0;
      numNodes = 1;
#endif
      if(mAccessorProjectionFunctor == NULL) {
        mAccessorProjectionFunctor = new AccessorProjectionFunctor*[numNodes];
      }
      mAccessorProjectionFunctor[nodeID] = functor;
    }
    
    ImageReduction::AccessorProjectionFunctor* ImageReduction::retrieveMyAccessorProjectionFunctor(int nodeID) {
#ifdef RUNNING_MULTINODE
      nodeID = 0;
#endif
      assert(mAccessorProjectionFunctor != NULL);
      return mAccessorProjectionFunctor[nodeID];
    }
    
    
    
    ////////////////
    
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
    
    
    
    static int compositeProjectionFunctorID(int numNodes, int numLevels, int level, int nodeID) {
      int accessorFunctorRangeMax = 1 + numNodes;
      int compositeFunctorIndex = nodeID * (numLevels + 1) + level;
      int result = accessorFunctorRangeMax + compositeFunctorIndex;
      return result;
    }
    
    
    static int numCompositeProjectionFunctorsPerNode(int maxTreeLevel) {
      return maxTreeLevel * maxTreeLevel / 2 + 1;
    }
    
    
    static int compositeProjectionFunctorIndex(int maxTreeLevel, int nodeID, int level) {
      int index = numCompositeProjectionFunctorsPerNode(maxTreeLevel) * (nodeID + 1) - (level + 1);
      return index;
    }
    
    
    ImageReduction::CompositeProjectionFunctor* ImageReduction::getCompositeProjectionFunctor(int nodeID, int maxTreeLevel, int level) {
      
      int index = compositeProjectionFunctorIndex(maxTreeLevel, nodeID, level);
      CompositeProjectionFunctor* result = mCompositeProjectionFunctor[index];
      return result;
    }
    
    
    
    ImageReduction::CompositeProjectionFunctor* ImageReduction::makeCompositeProjectionFunctor(int offset, int numBounds, int nodeID, int level, int numLevels, Runtime* runtime) {
      int id = compositeProjectionFunctorID(numBounds, numLevels, level, nodeID);
      CompositeProjectionFunctor* functor = new CompositeProjectionFunctor(offset, numBounds, id);
      runtime->register_projection_functor(id, functor);
      return functor;
    }
    
    
    
    static int accessorFunctorID(int nodeID) {
      return 1 + nodeID;
    }
    
    
    void ImageReduction::createProjectionFunctors(int nodeID, int numBounds, Runtime* runtime, ImageSize imageSize) {
      
      AccessorProjectionFunctor* accessorFunctor = new AccessorProjectionFunctor(mSimulationBounds, numBounds, accessorFunctorID(nodeID));
      runtime->register_projection_functor(accessorFunctorID(nodeID), accessorFunctor);
      storeMyAccessorProjectionFunctor(accessorFunctorID(nodeID), accessorFunctor, numBounds, nodeID);
      
      int numLevels = numTreeLevels(imageSize);
      if(mCompositeProjectionFunctor.empty()) {
        int numFunctors = numCompositeProjectionFunctorsPerNode(numLevels) * numBounds;
        mCompositeProjectionFunctor = std::vector<CompositeProjectionFunctor*>(numFunctors);
      }
      
      
      int offset = (int)powf(2.0f, numLevels - 1);
      for(int level = 0; level <= numLevels; level++) {
        CompositeProjectionFunctor* functor = makeCompositeProjectionFunctor(offset, numBounds, nodeID, level, numLevels, runtime);
        int index = compositeProjectionFunctorIndex(numLevels, nodeID, level);
        mCompositeProjectionFunctor[index] = functor;
        offset /= 2;
      }
    }
    
    
    
    
    
    
    
    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      ImageSize imageSize = ((ImageSize*)task->args)[0];
      SimulationBoundsCoordinate *domainBounds = (SimulationBoundsCoordinate*)((char*)task->args + sizeof(ImageSize));
      int numNodes = task->arglen / (fieldsPerSimulationBounds * sizeof(SimulationBoundsCoordinate));
      
      // this task initializes some per-node static state for use by subsequent tasks
      
      // set the node ID
      int nodeID = task->get_unique_id() % numNodes;
      storeMyNodeID(nodeID, numNodes);
      
      // load the bounds of the local simulation subdomain (axis aligned bounding box)
      // TODO do we just have to save a pointer here, or actually copy the data?
      SimulationBoundsCoordinate *myBounds = domainBounds + nodeID * fieldsPerSimulationBounds;
      storeMySimulationBounds(myBounds, nodeID, numNodes);
      
      // projection functors
      createProjectionFunctors(nodeID, numNodes, runtime, imageSize);
      
    }
    
    
    void ImageReduction::initializeNodes() {
      int size = sizeof(mSimulationBounds[0]) * mNumSimulationBounds * fieldsPerSimulationBounds;
      launch_task_by_depth(mInitialTaskID, mSimulationBounds, size, true);
    }
    
    
    void ImageReduction::initializeViewMatrix() {
      memset(mViewMatrix, 0, sizeof(mViewMatrix));
      mViewMatrix[0] = mViewMatrix[5] = mViewMatrix[10] = mViewMatrix[15] = 1.0f;
    }
    
    
    void ImageReduction::createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ImageSize imageSize) {
      mHierarchicalTreeDomain = std::vector<Domain>();
      Point<IMAGE_REDUCTION_DIMENSIONS> numFragments = imageSize.numFragments() - Point<IMAGE_REDUCTION_DIMENSIONS>::ONES();
      int numLeaves = 1;
      for(int level = 0; level < numTreeLevels; ++level) {
        numFragments.x[2] = numLeaves - 1;
        Rect<IMAGE_REDUCTION_DIMENSIONS> launchBounds(Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES(), numFragments);
        Domain domain = Domain::from_rect<IMAGE_REDUCTION_DIMENSIONS>(launchBounds);
        mHierarchicalTreeDomain.push_back(domain);
        numLeaves *= 2;
      }
      
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
                                                     PixelField *&r,
                                                     PixelField *&g,
                                                     PixelField *&b,
                                                     PixelField *&a,
                                                     PixelField *&z,
                                                     PixelField *&userdata,
                                                     ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS],
                                                     Runtime *runtime,
                                                     Context context) {
      
      Domain indexSpaceDomain = runtime->get_index_space_domain(context, region.get_logical_region().get_index_space());
      Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds = indexSpaceDomain.get_rect<IMAGE_REDUCTION_DIMENSIONS>();
      
      RegionAccessor<AccessorType::Generic, PixelField> acc_r, acc_g, acc_b, acc_a, acc_z, acc_userdata;
      
      createImageFieldPointer(acc_r, FID_FIELD_R, r, imageBounds, region, stride);
      createImageFieldPointer(acc_g, FID_FIELD_G, g, imageBounds, region, stride);
      createImageFieldPointer(acc_b, FID_FIELD_B, b, imageBounds, region, stride);
      createImageFieldPointer(acc_a, FID_FIELD_A, a, imageBounds, region, stride);
      createImageFieldPointer(acc_z, FID_FIELD_Z, z, imageBounds, region, stride);
      createImageFieldPointer(acc_userdata, FID_FIELD_USERDATA, userdata, imageBounds, region, stride);
    }
    
    
    FutureMap ImageReduction::launch_task_by_depth(unsigned taskID, void *args, int argLen, bool blocking){
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageSize) + argLen;
      char argsBuffer[totalArgLen];
      memcpy(argsBuffer, &mImageSize, sizeof(mImageSize));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageSize), args, argLen);
      }
      IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(argsBuffer, totalArgLen), argMap);
      RegionRequirement req(mDepthPartition, 0, READ_WRITE, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      depthLauncher.add_region_requirement(req);
      FutureMap futures = mRuntime->execute_index_space(mContext, depthLauncher);
      if(blocking) {
        futures.wait_all_results();
      }
      return futures;
    }
    
    
    
    
    void ImageReduction::composite_task(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime *runtime) {
#if NULL_COMPOSITE_TASKS
      return;//performance testing
#endif
      
      CompositeArguments args = ((CompositeArguments*)task->args)[0];
      PhysicalRegion fragment0 = regions[0];
      PhysicalRegion fragment1 = regions[1];
      
      std::cout << describe_task(task) << std::endl;
      
      ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
      PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
      PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
      ImageReductionComposite::CompositeFunction* compositeFunction;
      
      create_image_field_pointers(args.imageSize, fragment0, r0, g0, b0, a0, z0, userdata0, stride, runtime, ctx);
      create_image_field_pointers(args.imageSize, fragment1, r1, g1, b1, a1, z1, userdata1, stride, runtime, ctx);
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageSize.numPixelsPerFragment());
      
    }
    
    
    
    
    void ImageReduction::launchPipelineReduction(ScreenSpaceArguments args) {
      
    }
    
    
    
    
    FutureMap ImageReduction::launchTreeReduction(ImageSize imageSize, int treeLevel, int functorLevel, int offset,
                                                  GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination,
                                                  int compositeTaskID, LogicalPartition sourceFragmentPartition, LogicalRegion image,
                                                  Runtime* runtime, Context context,
                                                  int nodeID, int maxTreeLevel) {
      Domain launchDomain = mHierarchicalTreeDomain[treeLevel - 1];
      CompositeProjectionFunctor* functor0 = getCompositeProjectionFunctor(nodeID, maxTreeLevel, maxTreeLevel);
      CompositeProjectionFunctor* functor1 = getCompositeProjectionFunctor(nodeID, maxTreeLevel, treeLevel - 1);
      
      std::cout << "launch tree level " << treeLevel - 1 << " with functors " << functor0->to_string() << " and " << functor1->to_string() << " domain " << launchDomain << std::endl;
      
      ArgumentMap argMap;
      CompositeArguments args = { imageSize, depthFunc, blendFuncSource, blendFuncDestination };
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain, TaskArgument(&args, sizeof(args)), argMap);
      
      RegionRequirement req0(sourceFragmentPartition, functor0->id(), READ_WRITE, EXCLUSIVE, image);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);
      
      RegionRequirement req1(sourceFragmentPartition, functor1->id(), READ_ONLY, SIMULTANEOUS, image);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);
      
      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);
      
      if(treeLevel > 1) {
        futures = launchTreeReduction(imageSize, treeLevel - 1, functorLevel + 1, offset * 2, depthFunc, blendFuncSource, blendFuncDestination, compositeTaskID, sourceFragmentPartition, image, runtime, context, nodeID, maxTreeLevel);
      }
      return futures;
      
    }
    
    
    // this task sorts the data from the source image into the scratch image
    
    void ImageReduction::accessor_task(const Task *task,
                                       const std::vector<PhysicalRegion> &regions,
                                       Context ctx, HighLevelRuntime *runtime) {
      
      ImageSize imageSize = ((ImageSize*)task->args)[0];
      PhysicalRegion source = regions[0];
      PhysicalRegion destination = regions[1];
      
      Domain sourceDomain = runtime->get_index_space_domain(ctx, source.get_logical_region().get_index_space());
      Domain destinationDomain = runtime->get_index_space_domain(ctx, destination.get_logical_region().get_index_space());
      std::cout << describe_task(task) << " copy fragment from source " << sourceDomain.get_rect<IMAGE_REDUCTION_DIMENSIONS>() << " to scratch " << destinationDomain.get_rect<IMAGE_REDUCTION_DIMENSIONS>() << std::endl;
      
      ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS];
      PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
      PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
      ImageReductionComposite::CompositeFunction* compositeFunction;
      
      create_image_field_pointers(imageSize, source, r0, g0, b0, a0, z0, userdata0, stride, runtime, ctx);
      create_image_field_pointers(imageSize, destination, r1, g1, b1, a1, z1, userdata1, stride, runtime, ctx);
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(GL_ALWAYS, 0, 0);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r1, g1, b1, a1, z1, userdata1, imageSize.numPixelsPerFragment());
    }
    
    
    FutureMap ImageReduction::launchAccessorTasks() {
      ArgumentMap argMap;
      IndexTaskLauncher accessorLauncher(mAccessorTaskID, mSourceFragmentDomain, TaskArgument(&mImageSize, sizeof(mImageSize)), argMap);
      
      AccessorProjectionFunctor *functor = retrieveMyAccessorProjectionFunctor(accessorFunctorID(mLocalCopyOfNodeID));
      
      std::cout << "launch accessors localNodeID " << mLocalCopyOfNodeID
      << " functor id " << functor->id() << std::endl;
      
      RegionRequirement sourceRequirement(mSourceFragmentPartition, functor->id(), READ_ONLY, SIMULTANEOUS, mSourceImage);
      addImageFieldsToRequirement(sourceRequirement);
      accessorLauncher.add_region_requirement(sourceRequirement);
      
      RegionRequirement scratchRequirement(mScratchFragmentPartition, 0, WRITE_DISCARD, EXCLUSIVE, mScratchImage);
      addImageFieldsToRequirement(scratchRequirement);
      accessorLauncher.add_region_requirement(scratchRequirement);
      
      FutureMap futures = mRuntime->execute_index_space(mContext, accessorLauncher);
      return futures;
      
    }
    
    
    FutureMap ImageReduction::reduceAssociative() {
      int maxTreeLevel = numTreeLevels(mImageSize);
      int offset = 1;
      
      std::cout << "reduce associative node " << mLocalCopyOfNodeID << std::endl;
      assert(mLocalCopyOfNodeID >= 0 && mLocalCopyOfNodeID < 4);
      
      return launchTreeReduction(mImageSize, maxTreeLevel, 0, offset, mDepthFunction, mBlendFunctionSource, mBlendFunctionDestination,
                                 mCompositeTaskID, mSourceFragmentPartition, mSourceImage,
                                 mRuntime, mContext, mLocalCopyOfNodeID, maxTreeLevel);
    }
    
    FutureMap ImageReduction::reduce_associative_commutative(){
      return reduceAssociative();
    }
    
    FutureMap ImageReduction::reduce_associative_noncommutative(){
      return reduceAssociative();
    }
    
    
    FutureMap ImageReduction::reduceNonassociative() {
      //      return launchScreenSpaceTasks(false);
    }
    
    FutureMap ImageReduction::reduce_nonassociative_commutative(){
      return reduceNonassociative();
    }
    
    FutureMap ImageReduction::reduce_nonassociative_noncommutative(){
      return reduceNonassociative();
    }
    
    //this matrix project pixels from eye space bounded by l,r,t,p,n,f to clip space -1,1
    // for an index task launch we mwant an ffine map from a domain launch index to a subregions index in the partition
    
    static ImageReduction::SimulationBoundsCoordinate *homogeneousOrthographicProjectionMatrix(ImageReduction::SimulationBoundsCoordinate right,
                                                                                               ImageReduction::SimulationBoundsCoordinate left,
                                                                                               ImageReduction::SimulationBoundsCoordinate top,
                                                                                               ImageReduction::SimulationBoundsCoordinate bottom,
                                                                                               ImageReduction::SimulationBoundsCoordinate far,
                                                                                               ImageReduction::SimulationBoundsCoordinate near) {
      static ImageReduction::SimulationBoundsCoordinate *result = NULL;
      if(result == NULL) {
        result = new ImageReduction::SimulationBoundsCoordinate[16];
        // row major
        result[0] = (2.0f / (right - left));
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        //
        result[4] = 0.0f;
        result[5] = (2.0f / (top - bottom));
        result[6] = 0.0f;
        result[7] = 0.0f;
        //
        result[8] = 0.0f;
        result[9] = 0.0f;
        result[10] = (-2.0f / (far - near));
        result[11] = 0.0f;
        //
        result[12] = -((right + left) / (right - left));
        result[13] = -((top + bottom) / (top - bottom));
        result[14] = -((far + near) / (far - near));
        result[15] = 1.0f;
      }
      return result;
    }
    
#if 0
    static ImageReduction::SimulationBoundsCoordinate *inverseOrthographicProjectionMatrix(ImageReduction::SimulationBoundsCoordinate right,
                                                                                           ImageReduction::SimulationBoundsCoordinate left,
                                                                                           ImageReduction::SimulationBoundsCoordinate top,
                                                                                           ImageReduction::SimulationBoundsCoordinate bottom,
                                                                                           ImageReduction::SimulationBoundsCoordinate near,
                                                                                           ImageReduction::SimulationBoundsCoordinate far) {
      static ImageReduction::SimulationBoundsCoordinate *result = NULL;
      if(result == NULL) {
        result = new ImageReduction::SimulationBoundsCoordinate[16];
        ImageReduction::SimulationBoundsCoordinate* matrix = homogeneousOrthographicProjectionMatrix(right, left, top, bottom, near, far);
        
        // copied from MESA implementation of gluInvertMatrix
        // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
        
        ImageReduction::SimulationBoundsCoordinate det;
        int i;
        
        result[0] = matrix[5]  * matrix[10] * matrix[15] -
        matrix[5]  * matrix[11] * matrix[14] -
        matrix[9]  * matrix[6]  * matrix[15] +
        matrix[9]  * matrix[7]  * matrix[14] +
        matrix[13] * matrix[6]  * matrix[11] -
        matrix[13] * matrix[7]  * matrix[10];
        
        result[4] = -matrix[4]  * matrix[10] * matrix[15] +
        matrix[4]  * matrix[11] * matrix[14] +
        matrix[8]  * matrix[6]  * matrix[15] -
        matrix[8]  * matrix[7]  * matrix[14] -
        matrix[12] * matrix[6]  * matrix[11] +
        matrix[12] * matrix[7]  * matrix[10];
        
        result[8] = matrix[4]  * matrix[9] * matrix[15] -
        matrix[4]  * matrix[11] * matrix[13] -
        matrix[8]  * matrix[5] * matrix[15] +
        matrix[8]  * matrix[7] * matrix[13] +
        matrix[12] * matrix[5] * matrix[11] -
        matrix[12] * matrix[7] * matrix[9];
        
        result[12] = -matrix[4]  * matrix[9] * matrix[14] +
        matrix[4]  * matrix[10] * matrix[13] +
        matrix[8]  * matrix[5] * matrix[14] -
        matrix[8]  * matrix[6] * matrix[13] -
        matrix[12] * matrix[5] * matrix[10] +
        matrix[12] * matrix[6] * matrix[9];
        
        result[1] = -matrix[1]  * matrix[10] * matrix[15] +
        matrix[1]  * matrix[11] * matrix[14] +
        matrix[9]  * matrix[2] * matrix[15] -
        matrix[9]  * matrix[3] * matrix[14] -
        matrix[13] * matrix[2] * matrix[11] +
        matrix[13] * matrix[3] * matrix[10];
        
        result[5] = matrix[0]  * matrix[10] * matrix[15] -
        matrix[0]  * matrix[11] * matrix[14] -
        matrix[8]  * matrix[2] * matrix[15] +
        matrix[8]  * matrix[3] * matrix[14] +
        matrix[12] * matrix[2] * matrix[11] -
        matrix[12] * matrix[3] * matrix[10];
        
        result[9] = -matrix[0]  * matrix[9] * matrix[15] +
        matrix[0]  * matrix[11] * matrix[13] +
        matrix[8]  * matrix[1] * matrix[15] -
        matrix[8]  * matrix[3] * matrix[13] -
        matrix[12] * matrix[1] * matrix[11] +
        matrix[12] * matrix[3] * matrix[9];
        
        result[13] = matrix[0]  * matrix[9] * matrix[14] -
        matrix[0]  * matrix[10] * matrix[13] -
        matrix[8]  * matrix[1] * matrix[14] +
        matrix[8]  * matrix[2] * matrix[13] +
        matrix[12] * matrix[1] * matrix[10] -
        matrix[12] * matrix[2] * matrix[9];
        
        result[2] = matrix[1]  * matrix[6] * matrix[15] -
        matrix[1]  * matrix[7] * matrix[14] -
        matrix[5]  * matrix[2] * matrix[15] +
        matrix[5]  * matrix[3] * matrix[14] +
        matrix[13] * matrix[2] * matrix[7] -
        matrix[13] * matrix[3] * matrix[6];
        
        result[6] = -matrix[0]  * matrix[6] * matrix[15] +
        matrix[0]  * matrix[7] * matrix[14] +
        matrix[4]  * matrix[2] * matrix[15] -
        matrix[4]  * matrix[3] * matrix[14] -
        matrix[12] * matrix[2] * matrix[7] +
        matrix[12] * matrix[3] * matrix[6];
        
        result[10] = matrix[0]  * matrix[5] * matrix[15] -
        matrix[0]  * matrix[7] * matrix[13] -
        matrix[4]  * matrix[1] * matrix[15] +
        matrix[4]  * matrix[3] * matrix[13] +
        matrix[12] * matrix[1] * matrix[7] -
        matrix[12] * matrix[3] * matrix[5];
        
        result[14] = -matrix[0]  * matrix[5] * matrix[14] +
        matrix[0]  * matrix[6] * matrix[13] +
        matrix[4]  * matrix[1] * matrix[14] -
        matrix[4]  * matrix[2] * matrix[13] -
        matrix[12] * matrix[1] * matrix[6] +
        matrix[12] * matrix[2] * matrix[5];
        
        result[3] = -matrix[1] * matrix[6] * matrix[11] +
        matrix[1] * matrix[7] * matrix[10] +
        matrix[5] * matrix[2] * matrix[11] -
        matrix[5] * matrix[3] * matrix[10] -
        matrix[9] * matrix[2] * matrix[7] +
        matrix[9] * matrix[3] * matrix[6];
        
        result[7] = matrix[0] * matrix[6] * matrix[11] -
        matrix[0] * matrix[7] * matrix[10] -
        matrix[4] * matrix[2] * matrix[11] +
        matrix[4] * matrix[3] * matrix[10] +
        matrix[8] * matrix[2] * matrix[7] -
        matrix[8] * matrix[3] * matrix[6];
        
        result[11] = -matrix[0] * matrix[5] * matrix[11] +
        matrix[0] * matrix[7] * matrix[9] +
        matrix[4] * matrix[1] * matrix[11] -
        matrix[4] * matrix[3] * matrix[9] -
        matrix[8] * matrix[1] * matrix[7] +
        matrix[8] * matrix[3] * matrix[5];
        
        result[15] = matrix[0] * matrix[5] * matrix[10] -
        matrix[0] * matrix[6] * matrix[9] -
        matrix[4] * matrix[1] * matrix[10] +
        matrix[4] * matrix[2] * matrix[9] +
        matrix[8] * matrix[1] * matrix[6] -
        matrix[8] * matrix[2] * matrix[5];
        
        det = matrix[0] * result[0] + matrix[1] * result[4] + matrix[2] * result[8] + matrix[3] * result[12];
        
        assert(det != 0);//singularity - should never happen
        
        det = 1.0 / det;
        
        for (i = 0; i < 16; i++)
          result[i] = result[i] * det;
      }
      return result;
    }
#endif
    
    static inline int index(int row, int column) {
      // row major
      return row * 4 + column;
    }
    
    static void matrixMultiply4x4(ImageReduction::SimulationBoundsCoordinate* A,
                                  ImageReduction::SimulationBoundsCoordinate* B,
                                  ImageReduction::SimulationBoundsCoordinate* C) {
      // C = A x B
      // https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
      for (unsigned int i = 0; i < 16; i += 4)
        for (unsigned int j = 0; j < 4; ++j)
          C[i + j] = (B[i + 0] * A[j +  0])
          + (B[i + 1] * A[j +  4])
          + (B[i + 2] * A[j +  8])
          + (B[i + 3] * A[j + 12]);
    }
    
    
    // this returns the composite order for a simulation subdomain
    int ImageReduction::subdomainToCompositeIndex(SimulationBoundsCoordinate *bounds, int scale) {
      
      SimulationBoundsCoordinate l0 = bounds[0]; // xmin
      SimulationBoundsCoordinate b0 = bounds[1]; // ymin
      SimulationBoundsCoordinate n0 = bounds[2]; // zmin
      SimulationBoundsCoordinate r0 = bounds[3]; // xmax
      SimulationBoundsCoordinate t0 = bounds[4]; // ymax
      SimulationBoundsCoordinate f0 = bounds[5]; // zmax
      
      SimulationBoundsCoordinate center[] = { (r0 + l0) / 2.0f, (t0 + b0) / 2.0f, (f0 + n0) / 2.0f, 1.0f };
      SimulationBoundsCoordinate projected[] = { 0.0f, 0.0f, 0.0f, 0.0f };
      const int numHomgeneousCoordinates = 4;
      SimulationBoundsCoordinate *projection = homogeneousOrthographicProjectionMatrix(mXMax, mXMin, mYMax, mYMin, mZMax, mZMin);
      SimulationBoundsCoordinate viewProjection[16];
      matrixMultiply4x4(projection, mViewMatrix, viewProjection);
      
      // apply view and projection matrices to center of this fragment
      for(int i = 0; i < numHomgeneousCoordinates; ++i) {
        for(int j = 0; j < numHomgeneousCoordinates; ++j) {
          projected[i] += center[i] * viewProjection[index(i, j)];
        }
      }
      SimulationBoundsCoordinate z = projected[2] / projected[3];//range -1 to 1
      int scaledZ = (z + 1.0f) * (scale / 2);
      
      
      std::cout << " projected from center " << center[0] << "," << center[1] << "," << center[2] << "," << center[3] << " yields "
      << projected[0] << "," << projected[1] << "," << projected[2] << "," << projected[3]
      << " scaledZ is " << scaledZ << std::endl;
      
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
      create_image_field_pointers(args.imageSize, displayPlane, r, g, b, a, z, userdata, stride, runtime, ctx);
      
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
      RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      taskLauncher.add_region_requirement(req);
      Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
      return displayFuture;
    }
    
  }
}
