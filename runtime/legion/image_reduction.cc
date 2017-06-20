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
#include "image_reduction_composite.h"

#include <iostream>
#include <fstream>
#include <math.h>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
  namespace Visualization {
    
    
    // declare module static data
    
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
    std::vector<ImageReduction::CompositeProjectionFunctor*> *ImageReduction::mCompositeProjectionFunctor = NULL;
    int ImageReduction::mNodeCount;
    std::vector<Domain> *ImageReduction::mHierarchicalTreeDomain = NULL;
    GLfloat ImageReduction::mGlViewTransform[numMatrixElements4x4];
    ImageReduction::PixelField ImageReduction::mGlConstantColor[numPixelFields];
    GLenum ImageReduction::mGlBlendEquation;
    GLenum ImageReduction::mGlBlendFunctionSource;
    GLenum ImageReduction::mGlBlendFunctionDestination;
    
#include <unistd.h>//debug
    
    ImageReduction::ImageReduction(ImageSize imageSize, Context context, HighLevelRuntime *runtime) {
      mImageSize = imageSize;
      mContext = context;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mMySimulationBounds = NULL;
      mNodeID = NULL;
      
      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      
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
        mMySimulationBounds = NULL;
      }
      if(mHierarchicalTreeDomain != NULL) {
        delete mHierarchicalTreeDomain;
        mHierarchicalTreeDomain = NULL;
      }
      if(mCompositeProjectionFunctor != NULL) {
        delete mCompositeProjectionFunctor;
        mCompositeProjectionFunctor = NULL;
      }
      
      mRuntime->destroy_index_space(mContext, mSourceImage.get_index_space());
      mRuntime->destroy_logical_region(mContext, mSourceImage);
      mRuntime->destroy_index_partition(mContext, mDepthPartition.get_index_partition());
      mRuntime->destroy_logical_partition(mContext, mDepthPartition);
      mRuntime->destroy_index_partition(mContext, mSourceFragmentPartition.get_index_partition());
      mRuntime->destroy_logical_partition(mContext, mSourceFragmentPartition);
    }
    
    
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
      Rect<image_region_dimensions> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<image_region_dimensions>::ONES());
      domain = Domain::from_rect<image_region_dimensions>(imageBounds);
      IndexSpace pixels = mRuntime->create_index_space(mContext, domain);
      FieldSpace fields = imageFields();
      region = mRuntime->create_logical_region(mContext, pixels, fields);
    }
    
    
    void ImageReduction::partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      Blockify<image_region_dimensions> coloring(mImageSize.layerSize());
      IndexPartition imageDepthIndexPartition = mRuntime->create_index_partition(mContext, image.get_index_space(), coloring);
      partition = mRuntime->get_logical_partition(mContext, image, imageDepthIndexPartition);
      mRuntime->attach_name(partition, "image depth partition");
      Rect<image_region_dimensions> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<image_region_dimensions>::ONES());
      domain = Domain::from_rect<image_region_dimensions>(depthBounds);
    }
    
    
    void ImageReduction::partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      Blockify<image_region_dimensions> coloring(mImageSize.fragmentSize());
      IndexPartition imageFragmentIndexPartition = mRuntime->create_index_partition(mContext, image.get_index_space(), coloring);
      mRuntime->attach_name(imageFragmentIndexPartition, "image fragment index");
      partition = mRuntime->get_logical_partition(mContext, image, imageFragmentIndexPartition);
      mRuntime->attach_name(partition, "image fragment partition");
      Rect<image_region_dimensions> fragmentBounds(mImageSize.origin(), mImageSize.numFragments() - Point<image_region_dimensions>::ONES());
      domain = Domain::from_rect<image_region_dimensions>(fragmentBounds);
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
    
    
    
    
    
    ////////////////
    
    int ImageReduction::numTreeLevels(ImageSize imageSize) {
      int numTreeLevels = log2f(imageSize.numImageLayers);
      if(powf(2.0f, numTreeLevels) < imageSize.numImageLayers) {
        numTreeLevels++;
      }
      return numTreeLevels;
    }
    
    int ImageReduction::subtreeHeight(ImageSize imageSize) {
      const int totalLevels = numTreeLevels(imageSize);
      const int MAX_LEVELS_PER_SUBTREE = 7; // 128 tasks per subtree
      return (totalLevels < MAX_LEVELS_PER_SUBTREE) ? totalLevels : MAX_LEVELS_PER_SUBTREE;
    }
    
    
    
    void ImageReduction::createProjectionFunctors(int nodeID, int numBounds, Runtime* runtime, ImageSize imageSize) {
      
      if(mCompositeProjectionFunctor == NULL) {
        mCompositeProjectionFunctor = new std::vector<CompositeProjectionFunctor*>();
      }
      
      int numLevels = numTreeLevels(imageSize);
      
      for(int level = 0; level <= numLevels; ++level) {
        if(level >= mCompositeProjectionFunctor->size()) {
          int offset = (level == 0) ? 0 : (int)powf(2.0f, level - 1);
          ProjectionID id = runtime->generate_dynamic_projection_id();
          CompositeProjectionFunctor* functor = new CompositeProjectionFunctor(offset, numBounds, id);
          runtime->register_projection_functor(id, functor);
          mCompositeProjectionFunctor->push_back(functor);
        }
      }
    }
    
    
    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      ImageSize imageSize = ((ImageSize*)task->args)[0];
      SimulationBoundsCoordinate *domainBounds = (SimulationBoundsCoordinate*)((char*)task->args + sizeof(ImageSize));
      int numNodes = imageSize.numImageLayers;
      
      // this task initializes some per-node static state for use by subsequent tasks
      
      // set the node ID
      Domain indexSpaceDomain = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds = indexSpaceDomain.get_rect<image_region_dimensions>();
      
      int nodeID = imageBounds.lo.x[2];//TODO abstract the use of [2] throughout this code
      storeMyNodeID(nodeID, numNodes);
      
      // load the bounds of the local simulation subdomain (axis aligned bounding box)
      // TODO do we just have to save a pointer here, or actually copy the data?
      if(domainBounds != NULL) {
        SimulationBoundsCoordinate *myBounds = domainBounds + nodeID * fieldsPerSimulationBounds;
        storeMySimulationBounds(myBounds, nodeID, numNodes);
      }
      
      // projection functors
      createProjectionFunctors(nodeID, numNodes, runtime, imageSize);
      
    }
    
    
    void ImageReduction::initializeNodes() {
      int size = sizeof(mSimulationBounds[0]) * mNumSimulationBounds * fieldsPerSimulationBounds;
      launch_task_by_depth(mInitialTaskID, mSimulationBounds, size, true);
    }
    
    
    void ImageReduction::initializeViewMatrix() {
      memset(mGlViewTransform, 0, sizeof(mGlViewTransform));
      mGlViewTransform[0] = mGlViewTransform[5] = mGlViewTransform[10] = mGlViewTransform[15] = 1.0f;
    }
    
    
    void ImageReduction::createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ImageSize imageSize) {
      if(mHierarchicalTreeDomain == NULL) {
        mHierarchicalTreeDomain = new std::vector<Domain>();
      }
      
      Point<image_region_dimensions> numFragments = imageSize.numFragments() - Point<image_region_dimensions>::ONES();
      int numLeaves = 1;
      
      for(int level = 0; level < numTreeLevels; ++level) {
        if(level >= mHierarchicalTreeDomain->size()) {
          numFragments.x[2] = numLeaves - 1;
          Rect<image_region_dimensions> launchBounds(Point<image_region_dimensions>::ZEROES(), numFragments);
          Domain domain = Domain::from_rect<image_region_dimensions>(launchBounds);
          mHierarchicalTreeDomain->push_back(domain);
        }
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
    
    
    void ImageReduction::createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc,
                                                 int fieldID,
                                                 PixelField *&field,
                                                 Rect<image_region_dimensions> imageBounds,
                                                 PhysicalRegion region,
                                                 ByteOffset offset[image_region_dimensions]) {
      acc = region.get_field_accessor(fieldID).typeify<PixelField>();
      Rect<image_region_dimensions> tempBounds;
      field = acc.raw_rect_ptr<image_region_dimensions>(imageBounds, tempBounds, offset);
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
                                                     Stride stride,
                                                     Runtime *runtime,
                                                     Context context) {
      
      Domain indexSpaceDomain = runtime->get_index_space_domain(context, region.get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds = indexSpaceDomain.get_rect<image_region_dimensions>();
      
      RegionAccessor<AccessorType::Generic, PixelField> acc_r, acc_g, acc_b, acc_a, acc_z, acc_userdata;
      
      createImageFieldPointer(acc_r, FID_FIELD_R, r, imageBounds, region, stride[FID_FIELD_R]);
      createImageFieldPointer(acc_g, FID_FIELD_G, g, imageBounds, region, stride[FID_FIELD_G]);
      createImageFieldPointer(acc_b, FID_FIELD_B, b, imageBounds, region, stride[FID_FIELD_B]);
      createImageFieldPointer(acc_a, FID_FIELD_A, a, imageBounds, region, stride[FID_FIELD_A]);
      createImageFieldPointer(acc_z, FID_FIELD_Z, z, imageBounds, region, stride[FID_FIELD_Z]);
      createImageFieldPointer(acc_userdata, FID_FIELD_USERDATA, userdata, imageBounds, region, stride[FID_FIELD_USERDATA]);
    }
    
    
    FutureMap ImageReduction::launch_task_by_depth(unsigned taskID, void *args, int argLen, bool blocking){
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageSize) + argLen;
      char *argsBuffer = new char[totalArgLen];
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
      delete [] argsBuffer;
      return futures;
    }
    
    
    
#ifdef DEBUG
    static void dumpImage(ImageReduction::PixelField *rr, ImageReduction::PixelField*gg, ImageReduction::PixelField*bb, ImageReduction::PixelField*aa, ImageReduction::PixelField*zz, ImageReduction::PixelField*uu, ImageReduction::Stride stride, char *text) {
      std::cout << std::endl;
      std::cout << text << std::endl;
      for(int i = 0; i < 10; ++i) {
        std::cout << text << " pixel " << i << ": ";
        std::cout << rr[0] << "\t" << gg[0] << "\t" << bb[0] << "\t" << aa[0] << "\t" << zz[0] << "\t" << uu[0] << std::endl;
        ImageReductionComposite::increment(rr, gg, bb, aa, zz, uu, stride);
      }
    }
#endif
    
    
    
    
    void ImageReduction::composite_task(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime *runtime) {
#if NULL_COMPOSITE_TASKS
      return;//performance testing
#endif
      
      CompositeArguments args = ((CompositeArguments*)task->args)[0];
      PhysicalRegion fragment0 = regions[0];
      PhysicalRegion fragment1 = regions[1];
      
      Stride stride;
      PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
      PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
      ImageReductionComposite::CompositeFunction* compositeFunction;
      
//      UsecTimer composite("composite time:");
//      composite.start();
      create_image_field_pointers(args.imageSize, fragment0, r0, g0, b0, a0, z0, userdata0, stride, runtime, ctx);
      create_image_field_pointers(args.imageSize, fragment1, r1, g1, b1, a1, z1, userdata1, stride, runtime, ctx);
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageSize.numPixelsPerFragment(), stride);
//      composite.stop();
//      std::cout << composite.to_string() << std::endl;
    }
    
    
    
    
    
    FutureMap ImageReduction::launchTreeReduction(ImageSize imageSize, int treeLevel, int functorLevel, int offset,
                                                  GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                                  int compositeTaskID, LogicalPartition sourceFragmentPartition, LogicalRegion image,
                                                  Runtime* runtime, Context context,
                                                  int nodeID, int maxTreeLevel) {
      
      Domain launchDomain = (*mHierarchicalTreeDomain)[treeLevel - 1];
      CompositeProjectionFunctor* functor0 = (*mCompositeProjectionFunctor)[0];
      CompositeProjectionFunctor* functor1 = (*mCompositeProjectionFunctor)[functorLevel];
      
      ArgumentMap argMap;
      CompositeArguments args = { imageSize, depthFunc, blendFuncSource, blendFuncDestination, blendEquation };
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain, TaskArgument(&args, sizeof(args)), argMap);
      
      RegionRequirement req0(sourceFragmentPartition, functor0->id(), READ_WRITE, EXCLUSIVE, image);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);
      
      RegionRequirement req1(sourceFragmentPartition, functor1->id(), READ_ONLY, EXCLUSIVE, image);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);
      
      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);
      
      if(treeLevel > 1) {
        futures = launchTreeReduction(imageSize, treeLevel - 1, functorLevel + 1, offset * 2, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, compositeTaskID, sourceFragmentPartition, image, runtime, context, nodeID, maxTreeLevel);
      }
      return futures;
      
    }
    
    
    
    FutureMap ImageReduction::reduceAssociative() {
      int maxTreeLevel = numTreeLevels(mImageSize);
      return launchTreeReduction(mImageSize, maxTreeLevel, 1, 1, mDepthFunction, mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
                                 mCompositeTaskID, mSourceFragmentPartition, mSourceImage,
                                 mRuntime, mContext, mLocalCopyOfNodeID, maxTreeLevel);
    }
    
    
    FutureMap ImageReduction::reduce_associative_commutative(){
      return reduceAssociative();
    }
    
    FutureMap ImageReduction::reduce_associative_noncommutative(){
      if(mNumSimulationBounds == mImageSize.numImageLayers) {
        return reduceAssociative();
      } else {
        std::cout << "cannot reduce noncommutatively until simulation bounds are provided" << std::endl;
        std::cout << "call preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) before startings Legion runtime" << std::endl;
        return FutureMap();
      }
    }
    
    
    
    FutureMap ImageReduction::launchPipelineReduction() {
      
    }
    
    
    
    FutureMap ImageReduction::reduceNonassociative() {
      return launchPipelineReduction();
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
        result = new ImageReduction::SimulationBoundsCoordinate[ImageReduction::numMatrixElements4x4];
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
        result = new ImageReduction::SimulationBoundsCoordinate[ImageReduction::numMatrixElements4x4];
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
        
        for (i = 0; i < numMatrixElements4x4; i++)
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
      SimulationBoundsCoordinate viewProjection[numMatrixElements4x4];
      matrixMultiply4x4(projection, mGlViewTransform, viewProjection);
      
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
      Stride stride;
      PixelField *r, *g, *b, *a, *z, *userdata;
      create_image_field_pointers(args.imageSize, displayPlane, r, g, b, a, z, userdata, stride, runtime, ctx);
      
      FILE *outputFile = fopen(outputFileName.c_str(), "wb");
      fwrite(r, numPixelFields * sizeof(*r), args.imageSize.pixelsPerLayer(), outputFile);
      fclose(outputFile);
      
      display.stop();
      cout << display.to_string() << endl;
    }
    
    
    
    Future ImageReduction::display(int t) {
      DisplayArguments args = { mImageSize, t };
      TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
      DomainPoint origin = DomainPoint::from_point<image_region_dimensions>(Point<image_region_dimensions>::ZEROES());
      LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mDepthPartition, origin);
      RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      taskLauncher.add_region_requirement(req);
      Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
      return displayFuture;
    }
    
  }
}
