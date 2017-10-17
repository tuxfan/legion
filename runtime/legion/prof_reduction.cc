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

#include "prof_reduction.h"

#include <iostream>
#include <fstream>
#include <math.h>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
  namespace Profile {
    
    
    // declare module static data
    
    int *ProfReduction::mNodeID;
    std::vector<ProfReduction::CompositeProjectionFunctor*> *ProfReduction::mCompositeProjectionFunctor = NULL;
    int ProfReduction::mNodeCount;
    std::vector<Domain> *ProfReduction::mHierarchicalTreeDomain = NULL;
    TaskID ProfReduction::mInitialTaskID;
    TaskID ProfReduction::mCompositeTaskID;
    
    
    ProfReduction::ProfReduction(ProfSize profSize, Context context, HighLevelRuntime *runtime) {
      mProfSize = profSize;
      mContext = context;
      mRuntime = runtime;
      mNodeID = NULL;
      
      createProfile(mSourceProfile, mSourceProfileDomain);
      partitionProfileByNode(mSourceProfile, mNodeDomain, mNodePartition);
      
      initializeNodes();
      assert(mNodeCount > 0);
      mLocalCopyOfNodeID = mNodeID[mNodeCount - 1];//written by initial_task
      createTreeDomains(mLocalCopyOfNodeID, numTreeLevels(profSize), runtime, profSize);
      
    }
    
    ProfReduction::~ProfReduction() {
      if(mHierarchicalTreeDomain != NULL) {
        delete mHierarchicalTreeDomain;
        mHierarchicalTreeDomain = NULL;
      }
      if(mCompositeProjectionFunctor != NULL) {
        delete mCompositeProjectionFunctor;
        mCompositeProjectionFunctor = NULL;
      }
      
      mRuntime->destroy_index_space(mContext, mSourceProfile.get_index_space());
      mRuntime->destroy_logical_region(mContext, mSourceProfile);
      mRuntime->destroy_index_partition(mContext, mNodePartition.get_index_partition());
      mRuntime->destroy_logical_partition(mContext, mNodePartition);
    }
    
    
    // this function should always be called prior to starting the Legion runtime

    void ProfReduction::initialize() {
      registerTasks();
    }
    
    // this function should be called prior to starting the Legion runtime
    // its purpose is to register tasks with the same id on all nodes
    
    void ProfReduction::registerTasks() {
      
      mInitialTaskID = Legion::HighLevelRuntime::generate_static_task_id();
      Legion::HighLevelRuntime::register_legion_task<initial_task>(mInitialTaskID,
                                                                   Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,                                                                                                  AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "initial_task");
      
      mCompositeTaskID = Legion::HighLevelRuntime::generate_static_task_id();
      Legion::HighLevelRuntime::register_legion_task<composite_task>(mCompositeTaskID,
                                                                     Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,                                                                                                  AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "composite_task");
      
    }
    
    
    FieldSpace ProfReduction::imageFields() {
      FieldSpace fields = mRuntime->create_field_space(mContext);
      mRuntime->attach_name(fields, "pixel fields");
      {
        FieldAllocator allocator = mRuntime->create_field_allocator(mContext, fields);
        FieldID fidr = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_R);
        assert(fidr == FID_FIELD_R);
        FieldID fidg = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_G);
        assert(fidg == FID_FIELD_G);
        FieldID fidb = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_B);
        assert(fidb == FID_FIELD_B);
        FieldID fida = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_A);
        assert(fida == FID_FIELD_A);
        FieldID fidz = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_Z);
        assert(fidz == FID_FIELD_Z);
        FieldID fidUserdata = allocator.allocate_field(sizeof(ProfileField), FID_FIELD_USERDATA);
        assert(fidUserdata == FID_FIELD_USERDATA);
      }
      return fields;
    }
    
    
    void ProfReduction::createProfile(LogicalRegion &region, Domain &domain) {
      Rect<image_region_dimensions> imageBounds(mProfSize.origin(), mProfSize.upperBound() - Point<image_region_dimensions>(1));
      domain = Domain(imageBounds);
      IndexSpace pixels = mRuntime->create_index_space(mContext, domain);
      FieldSpace fields = imageFields();
      region = mRuntime->create_logical_region(mContext, pixels, fields);
    }
    
    
    void ProfReduction::partitionProfileByNode(LogicalRegion prof, Domain &domain, LogicalPartition &partition) {
      IndexSpaceT<image_region_dimensions> parent(prof.get_index_space());
      Point<image_region_dimensions> blockingFactor = mProfSize.layerSize();
      IndexPartition imageDepthIndexPartition = mRuntime->create_partition_by_blockify(mContext, parent, blockingFactor);
      partition = mRuntime->get_logical_partition(mContext, prof, imageDepthIndexPartition);
      mRuntime->attach_name(partition, "prof depth partition");
      Rect<image_region_dimensions> depthBounds(mProfSize.origin(), mProfSize.numLayers() - Point<image_region_dimensions>(1));
      domain = Domain(depthBounds);
    }
    
  
    
    ///////////////
    //FIXME awkwardness about running multithreaded versus multinode can this be removed
    
    void ProfReduction::storeMyNodeID(int nodeID, int numNodes) {
      if(mNodeID == NULL) {
        mNodeID = new int[numNodes];
      }
      mNodeID[nodeID] = nodeID;
      mNodeCount++;
    }
    
    ////////////////
    
    
    
    int ProfReduction::numTreeLevels(ProfSize profSize) {
      int numTreeLevels = log2f(profSize.numImageLayers);
      if(powf(2.0f, numTreeLevels) < profSize.numImageLayers) {
        numTreeLevels++;
      }
      return numTreeLevels;
    }
    
    int ProfReduction::subtreeHeight(ProfSize profSize) {
      const int totalLevels = numTreeLevels(profSize);
      const int MAX_LEVELS_PER_SUBTREE = 10; // 1024 tasks per subtree
      return (totalLevels < MAX_LEVELS_PER_SUBTREE) ? totalLevels : MAX_LEVELS_PER_SUBTREE;
    }
    
    
    static int level2FunctorID(int level, int more) {
      return 100 + level * 2 + more;//TODO assign ids dynamically
    }
    
    
    void ProfReduction::createProjectionFunctors(int nodeID, int numBounds, Runtime* runtime, ProfSize profSize) {
      
      // really need a lock here on mCompositeProjectionFunctor when running multithreaded locally
      // not a problem for multinode runs
      if(mCompositeProjectionFunctor == NULL) {
        mCompositeProjectionFunctor = new std::vector<CompositeProjectionFunctor*>();
        
        int numLevels = numTreeLevels(profSize);
        int multiplier = profSize.numImageLayers;
        for(int level = 0; level < numLevels; ++level) {
          
          ProjectionID id0 = level2FunctorID(level, 0);
          int offset = 0;
          CompositeProjectionFunctor* functor0 = new CompositeProjectionFunctor(offset, multiplier, numBounds, id0);
          runtime->register_projection_functor(id0, functor0);
          mCompositeProjectionFunctor->push_back(functor0);

          ProjectionID id1 = level2FunctorID(level, 1);
          offset = multiplier / 2;
          CompositeProjectionFunctor* functor1 = new CompositeProjectionFunctor(offset, multiplier, numBounds, id1);
          runtime->register_projection_functor(id1, functor1);
          mCompositeProjectionFunctor->push_back(functor1);
          
          multiplier /= num_fragments_per_composite;
        }
      }
    }
    
    
    
    // this task and everything it calls is invoked on every node during initialization
    
    void ProfReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      
      ProfSize profSize = ((ProfSize*)task->args)[0];
      int numNodes = profSize.numImageLayers;
      
      // set the node ID
      Domain indexSpaceDomain = runtime->get_index_space_domain(regions[0].get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds = indexSpaceDomain;
      
      // get your node ID from the Z coordinate of your region
      int nodeID = imageBounds.lo[2];//TODO abstract the use of [2] throughout this code
      storeMyNodeID(nodeID, numNodes);
      
      // projection functors
      createProjectionFunctors(nodeID, numNodes, runtime, profSize);
    }
    
    
    void ProfReduction::initializeNodes() {
      launch_task_by_nodeID(mInitialTaskID, NULL, 0, true);
    }
    
    
    void ProfReduction::createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ProfSize profSize) {
      if(mHierarchicalTreeDomain == NULL) {
        mHierarchicalTreeDomain = new std::vector<Domain>();
      }
      
      Point<image_region_dimensions> numFragments = profSize.numFragments() - Point<image_region_dimensions>(1);
      int numLeaves = 1;
      
      for(int level = 0; level < numTreeLevels; ++level) {
        if((unsigned)level >= mHierarchicalTreeDomain->size()) {
          numFragments[2] = numLeaves - 1;
          Rect<image_region_dimensions> launchBounds(Point<image_region_dimensions>::ZEROES(), numFragments);
          Domain domain = Domain(launchBounds);
          mHierarchicalTreeDomain->push_back(domain);
        }
        numLeaves *= 2;
      }
      
    }
    
    
    void ProfReduction::addImageFieldsToRequirement(RegionRequirement &req) {
      req.add_field(FID_FIELD_R);
      req.add_field(FID_FIELD_G);
      req.add_field(FID_FIELD_B);
      req.add_field(FID_FIELD_A);
      req.add_field(FID_FIELD_Z);
      req.add_field(FID_FIELD_USERDATA);
    }
    
    
    void ProfReduction::createImageFieldPointer(RegionAccessor<AccessorType::Generic, ProfileField> &acc,
                                                 int fieldID,
                                                 ProfileField *&field,
                                                 Rect<image_region_dimensions> imageBounds,
                                                 PhysicalRegion region,
                                                 ByteOffset offset[image_region_dimensions]) {
      acc = region.get_field_accessor(fieldID).typeify<ProfileField>();
      LegionRuntime::Arrays::Rect<image_region_dimensions> tempBounds;
      LegionRuntime::Arrays::Rect<image_region_dimensions> bounds = Domain(imageBounds).get_rect<image_region_dimensions>();
      field = acc.raw_rect_ptr<image_region_dimensions>(bounds, tempBounds, offset);
      assert(bounds == tempBounds);
    }
    
    
    void ProfReduction::create_image_field_pointers(ProfSize profSize,
                                                     PhysicalRegion region,
                                                     ProfileField *&r,
                                                     ProfileField *&g,
                                                     ProfileField *&b,
                                                     ProfileField *&a,
                                                     ProfileField *&z,
                                                     ProfileField *&userdata,
                                                     Stride stride,
                                                     Runtime *runtime,
                                                     Context context) {
      
      Domain indexSpaceDomain = runtime->get_index_space_domain(context, region.get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds = indexSpaceDomain;
      
      RegionAccessor<AccessorType::Generic, ProfileField> acc_r, acc_g, acc_b, acc_a, acc_z, acc_userdata;
      
      createImageFieldPointer(acc_r, FID_FIELD_R, r, imageBounds, region, stride[FID_FIELD_R]);
      createImageFieldPointer(acc_g, FID_FIELD_G, g, imageBounds, region, stride[FID_FIELD_G]);
      createImageFieldPointer(acc_b, FID_FIELD_B, b, imageBounds, region, stride[FID_FIELD_B]);
      createImageFieldPointer(acc_a, FID_FIELD_A, a, imageBounds, region, stride[FID_FIELD_A]);
      createImageFieldPointer(acc_z, FID_FIELD_Z, z, imageBounds, region, stride[FID_FIELD_Z]);
      createImageFieldPointer(acc_userdata, FID_FIELD_USERDATA, userdata, imageBounds, region, stride[FID_FIELD_USERDATA]);
    }
    
    
    FutureMap ProfReduction::launch_task_by_nodeID(unsigned taskID, void *args, int argLen, bool blocking){
      
      ArgumentMap argMap;
      int totalArgLen = sizeof(mProfSize) + argLen;
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mProfSize, sizeof(mProfSize));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mProfSize), args, argLen);
      }
      IndexTaskLauncher depthLauncher(taskID, mNodeDomain, TaskArgument(argsBuffer, totalArgLen), argMap);
      RegionRequirement req(mNodePartition, 0, READ_WRITE, EXCLUSIVE, mSourceProfile);
      addImageFieldsToRequirement(req);
      depthLauncher.add_region_requirement(req);
      FutureMap futures = mRuntime->execute_index_space(mContext, depthLauncher);
      if(blocking) {
        futures.wait_all_results();
      }
      delete [] argsBuffer;
      return futures;
    }
    
    
    
    void ProfReduction::composite_task(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime *runtime) {
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif

      CompositeArguments args = ((CompositeArguments*)task->args)[0];
      ProfSize profSize = args.profSize;
      PhysicalRegion fragment0 = regions[0];
      PhysicalRegion fragment1 = regions[1];
      
      // here do the compositing
      
    }
    
    
    
    
    
    FutureMap ProfReduction::launchTreeReduction(ProfSize profSize, int treeLevel,
                                                   int compositeTaskID, LogicalPartition nodePartition, LogicalRegion prof,
                                                  Runtime* runtime, Context context,
                                                  int nodeID, int maxTreeLevel) {
      Domain launchDomain = (*mHierarchicalTreeDomain)[treeLevel - 1];
      int index = (treeLevel - 1) * 2;
      CompositeProjectionFunctor* functor0 = (*mCompositeProjectionFunctor)[index];
      CompositeProjectionFunctor* functor1 = (*mCompositeProjectionFunctor)[index + 1];
      
#if 0
      std::cout << " tree level " << treeLevel << " using functors " << functor0->to_string() << " " << functor1->to_string() << std::endl;
      std::cout << "launch domain at tree level " << treeLevel
      << launchDomain << std::endl;
#endif
      
      ArgumentMap argMap;
      CompositeArguments args = { profSize };
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain, TaskArgument(&args, sizeof(args)), argMap);
      
      RegionRequirement req0(nodePartition, functor0->id(), READ_WRITE, EXCLUSIVE, prof);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);
      
      RegionRequirement req1(nodePartition, functor1->id(), READ_ONLY, EXCLUSIVE, prof);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);
      
      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);
      
      if(treeLevel > 1) {
              
        futures = launchTreeReduction(profSize, treeLevel - 1, compositeTaskID, nodePartition, prof, runtime, context, nodeID, maxTreeLevel);
      }
      return futures;
      
    }
    
    
    
    FutureMap ProfReduction::reduceAssociative() {
      int maxTreeLevel = numTreeLevels(mProfSize);
      return launchTreeReduction(mProfSize, maxTreeLevel,
                                 mCompositeTaskID, mNodePartition, mSourceProfile,
                                 mRuntime, mContext, mLocalCopyOfNodeID, maxTreeLevel);
    }
    
    
    FutureMap ProfReduction::reduce_associative_commutative(){
      return reduceAssociative();
    }
    
    
    
  }
}
