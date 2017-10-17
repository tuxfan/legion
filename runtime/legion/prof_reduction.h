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



#ifndef prof_reduction_h
#define prof_reduction_h

//tracing -- remove this
//#define _T {std::cout<<__FILE__<<":"<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

#include "legion_visualization.h"

#include "usec_timer.h"
#include "accessor.h"


#include <iostream>
#include <sstream>




namespace Legion {
  namespace Profile {
    
    
    class ProfReduction {
      
    private:
      
      typedef struct {
        ProfSize profSize;
      } CompositeArguments;
      

    public:

      enum FieldIDs {
        FID_FIELD_R = 0,
        FID_FIELD_G,
        FID_FIELD_B,
        FID_FIELD_A,
        FID_FIELD_Z,
        FID_FIELD_USERDATA,
      };
      
      typedef float ProfileField;
      static const int numPixelFields = 6;//rgbazu
      static const int num_fragments_per_composite = 2;
      typedef ByteOffset Stride[ProfReduction::numPixelFields][image_region_dimensions];
      
      /**
       * Initialize the prof reduction framework.
       * Be sure to call this before starting the Legion runtime.
       */
      static void initialize();

      
      ProfReduction(){}
      /**
       * Construct an prof reduction framework.
       *
       * @param profSize defines dimensions of current prof
       * @param ctx Legion context
       * @param runtime  Legion runtime
       */
      ProfReduction(ProfSize profSize, Context ctx, HighLevelRuntime *runtime);
      /**
       * Destroy an instance of an prof reduction framework.
       */
      virtual ~ProfReduction();
      
      /**
       * Launch a set of tasks that each receive one layer in Z of the prof space.
       * Use this for example to render to the individual layers.
       *
       * @param taskID ID of task that has previously been registered with the Legion runtime
       */
      FutureMap launch_task_by_nodeID(unsigned taskID, void *args = NULL, int argLen = 0, bool blocking = false);
      /**
       * Perform a tree reduction using an associative commutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       */
      FutureMap reduce_associative_commutative();
      /**
       * Perform a tree reduction using an associative noncommutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       * Be sure to call preregisterSimulationBounds before starting the Legion runtime.
       */
      
      
      
      /**
       * obtain raw pointers to prof data
       *
       * @param profSize see legion_visualization.h
       * @param region physical region of prof fragment
       * @param r return raw pointer to pixel fields
       * @param g return raw pointer to pixel fields
       * @param b return raw pointer to pixel fields
       * @param a return raw pointer to pixel fields
       * @param z return raw pointer to pixel fields
       * @param userdata return raw pointer to pixel fields
       * @param stride returns stride between successive pixels
       */
      static void create_image_field_pointers(ProfSize profSize,
                                              PhysicalRegion region,
                                              ProfileField *&r,
                                              ProfileField *&g,
                                              ProfileField *&b,
                                              ProfileField *&a,
                                              ProfileField *&z,
                                              ProfileField *&userdata,
                                              Stride stride,
                                              Runtime *runtime,
                                              Context context);
      
      /**
       * Utility function to provide descriptive output for messages.
       *
       * @param task Legion task pointer
       */
      static std::string describe_task(const Task *task) {
        std::ostringstream output;
        output << task->get_task_name() << " " << task->get_unique_id()
        << " (" << task->index_point.point_data[0]
        << ", " << task->index_point.point_data[1]
        << ", " << task->index_point.point_data[2]
        << ")"
        ;
        return output.str();
      }
      
      static int numTreeLevels(ProfSize profSize);
      
      static void initial_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
      
      static void composite_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime);
      


    protected:
            
      class CompositeProjectionFunctor : public ProjectionFunctor {
      public:
        CompositeProjectionFunctor(int offset, int multiplier, int numBounds, int id) {
          mOffset = offset;
          mMultiplier = multiplier;
          mNumBounds = numBounds;
          mID = id;
        }
        
        virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                      LogicalPartition upperBound,
                                      const DomainPoint &point) {
          int launchDomainLayer = point[2];
          DomainPoint remappedPoint = point;
          int remappedLayer = launchDomainLayer * mMultiplier + mOffset;
          // handle non-power of 2 simulation size
          if(mNumBounds == 0 || remappedLayer < mNumBounds) {
            remappedPoint[2] = remappedLayer;
          }
          
#if 0
          {std::cout<< to_string() << " for task " << mappable->as_task()->get_unique_id()
            << " remaps launch point "<<point<<" to "<<remappedPoint<<std::endl;}
#endif
          
          LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, remappedPoint);
          return result;
        }
        
        int id() const{ return mID; }
        std::string to_string() const {
          char buffer[256];
          sprintf(buffer, "CompositeProjectionFunctor id %d offset %d multiplier %d numNodes %d", mID, mOffset, mMultiplier, mNumBounds);
          return std::string(buffer);
        }
        
        virtual bool is_exclusive(void) const{ return true; }
        virtual unsigned get_depth(void) const{ return 0; }
        
      private:
        int mOffset;
        int mMultiplier;
        int mNumBounds;
        int mID;
      };
      
      static CompositeProjectionFunctor* getCompositeProjectionFunctor(int nodeID, int maxTreeLevel, int level);

      static CompositeProjectionFunctor* makeCompositeProjectionFunctor(int offset, int numBounds, int nodeID, int level, int numLevels, Runtime* runtime);
      
      
       static void storeMyNodeID(int nodeID, int numNodes);
      
      static void createProjectionFunctors(int nodeID, int numBounds, Runtime* runtime, ProfSize profSize);
      
      
      
      void initializeNodes();
      void createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ProfSize mProfSize);
      FieldSpace imageFields();
      void createProfile(LogicalRegion &region, Domain &domain);
      void partitionProfileByNode(LogicalRegion prof, Domain &domain, LogicalPartition &partition);
      
      FutureMap reduceAssociative();
      
      void addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, ArgumentMap &argMap, int layer0, int layer1);
      
      void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);

      static void registerTasks();
      
      static void addImageFieldsToRequirement(RegionRequirement &req);
      
      
      static void createImageFieldPointer(LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic, ProfileField> &acc,
                                          int fieldID,
                                          ProfileField *&field,
                                          Rect<image_region_dimensions> imageBounds,
                                          PhysicalRegion region,
                                          ByteOffset offset[image_region_dimensions]);
      
      static int subtreeHeight(ProfSize profSize);
      
      static FutureMap launchTreeReduction(ProfSize profSize, int treeLevel,
                                           int compositeTaskID,
                                           LogicalPartition nodePartition, LogicalRegion prof,
                                           Runtime* runtime, Context context,
                                           int nodeID, int maxTreeLevel);
      
      
      ProfSize mProfSize;
      Context mContext;
      Runtime *mRuntime;
      LogicalPartition mNodePartition;
      LogicalRegion mSourceProfile;
      Domain mSourceProfileDomain;
      Domain mNodeDomain;
      Domain mCompositePipelineDomain;
      Domain mDisplayDomain;
      Domain mSourceFragmentDomain;
      int mAccessorFunctorID;
      int mLocalCopyOfNodeID;
      
    public:
      static int* mNodeID;
      static int mNodeCount;
      static std::vector<CompositeProjectionFunctor*> *mCompositeProjectionFunctor;
      static std::vector<Domain> *mHierarchicalTreeDomain;
      static TaskID mInitialTaskID;
      static TaskID mCompositeTaskID;
    };
    
  }
}

#endif /* prof_reduction_h */
