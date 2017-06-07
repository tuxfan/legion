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


#ifndef image_reduction_h
#define image_reduction_h

#include "legion_visualization.h"
#include "image_reduction_composite.h"

#include "usec_timer.h"

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <iostream>
#include <sstream>


namespace Legion {
  namespace Visualization {
    
    
    class ImageReduction {
      
    private:
      
      typedef struct {
        ImageSize imageSize;
        bool isAssociative;
        int compositeTaskID;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
        LogicalRegion image;
      } ScreenSpaceArguments;
      
      typedef struct {
        ImageSize imageSize;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
      } CompositeArguments;
      
      typedef struct {
        ImageSize imageSize;
        int t;
      } DisplayArguments;
      
      typedef std::vector<Future> FutureSet;
      
    public:
      
      typedef float SimulationBoundsCoordinate;
      static void preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBoundsX, int numBoundsY, int numBoundsZ);
      
      ImageReduction(){}
      /**
       * Construct an image reduction framework.
       *
       * @param imageSize defines dimensions of current image
       * @param ctx Legion context
       * @param runtime  Legion runtime
       */
      ImageReduction(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
      /**
       * Destroy an instance of an image reduction framework.
       */
      virtual ~ImageReduction();
      
      /**
       * Launch a set of tasks that each receive one layer in Z of the image space.
       * Use this for example to render to the individual layers.
       *
       * @param taskID ID of task that has previously been registered with the Legion runtime
       */
      FutureMap launch_task_by_depth(unsigned taskID, void *args = NULL, int argLen = 0, bool blocking = false);
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
      FutureMap reduce_associative_noncommutative();
      /**
       * Perform a pipeline reduction using a nonassociative commutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       */
      FutureMap reduce_nonassociative_commutative();
      /**
       * Perform a pipeline reduction using a nonassociative noncommutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       * Be sure to call preregisterSimulationBounds before starting the Legion runtime.
       */
      FutureMap reduce_nonassociative_noncommutative();
      /**
       * Move reduced image result to a display.
       *
       * @param t integer timestep
       */
      Future display(int t);
      
      /**
       * Define a blend operator to use in subsequent reductions.
       * For definition of blend factors see glBlendFunc (OpenGL).
       *
       * @param sfactor source blend factor
       * @param dfactor destination blend factor
       */
      void set_blend_func(GLenum sfactor, GLenum dfactor) {
        mBlendFunctionSource = sfactor;
        mBlendFunctionDestination = dfactor;
      }
      
      /**
       * Define a depth operator to use in subsequent reductions.
       * For definition of depth factors see glDepthFunc (OpenGL).
       * @param func depth comparison factor
       */
      void set_depth_func(GLenum func){ mDepthFunction = func; }
      
      /**
       * obtain raw pointers to image data
       *
       * @param imageSize see legion_visualization.h
       * @param region physical region of image fragment
       * @param r return raw pointer to pixel fields
       * @param g return raw pointer to pixel fields
       * @param b return raw pointer to pixel fields
       * @param a return raw pointer to pixel fields
       * @param z return raw pointer to pixel fields
       * @param userdata return raw pointer to pixel fields
       * @param stride returns stride between successive pixels
       */
      static void create_image_field_pointers(ImageSize imageSize,
                                              PhysicalRegion region,
                                              PixelField *&r,
                                              PixelField *&g,
                                              PixelField *&b,
                                              PixelField *&a,
                                              PixelField *&z,
                                              PixelField *&userdata,
                                              ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS],
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
      
      static void display_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
      
    private:
      
      class AccessorProjectionFunctor : public ProjectionFunctor {
      public:
        AccessorProjectionFunctor(SimulationBoundsCoordinate* bounds, int numBounds, int functorID){
          mBounds = bounds;
          mNumBounds = numBounds;
          mID = functorID;
        }
        
        virtual LogicalRegion project(Context context, Task *task,
                                      unsigned index,
                                      LogicalPartition upperBound,
                                      const DomainPoint &point) {
          LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(context, upperBound, newPoint(point));
          return result;
        }
        
        int id() const{ return mID; }
        
        virtual bool is_exclusive(void) const{ return true; }
        virtual unsigned get_depth(void) const{ return 0; }
        
      private:
        SimulationBoundsCoordinate* mBounds;
        int mNumBounds;
        int mID;
        
        DomainPoint newPoint(DomainPoint point) {
          int nodeID = point[2];
          SimulationBoundsCoordinate* bounds = mBounds + nodeID * fieldsPerSimulationBounds;
          
          std::cout << "point " << point << " bounds in functor " << mID << " are " << bounds[0] << "," << bounds[1] << "," << bounds[2]
          << " to " << bounds[3] << "," << bounds[4] << "," << bounds[5] << std::endl;
          
          point[2] = subdomainToCompositeIndex(bounds, mNumBounds);
          return point;
        }
      };
      
      
      static void storeMySimulationBounds(SimulationBoundsCoordinate* values, int nodeID, int numNodes);
      
      static SimulationBoundsCoordinate* retrieveMySimulationBounds(int nodeID);
      
      static void storeMyNodeID(int nodeID, int numNodes);
      
      static int retrieveMyNodeID(int nodeID);
      
      static void storeMyProjectionFunctor(int functorID, AccessorProjectionFunctor* functor, int numNodes, int nodeID);
      
      static AccessorProjectionFunctor* retrieveMyProjectionFunctor(int nodeID);
      
      static void createProjectionFunctor(int nodeID, int numBounds, Runtime* runtime);
      
      static void initial_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
      
      static void accessor_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime);
      
      static void composite_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime);
      
      

      
      void initializeNodes();
      FieldSpace imageFields();
      void createImage(LogicalRegion &region, Domain &domain);
      void partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition);
      void partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition);
      
      FutureMap launchScreenSpaceTasks(bool isAssociative);
      FutureMap launchAccessorTasks();
      
      FutureMap reduceAssociative();
      FutureMap reduceNonassociative();
      
      void addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, ArgumentMap &argMap, int layer0, int layer1);
      
      void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);
      int *defaultPermutation();
      void registerTasks();
      
      static void addImageFieldsToRequirement(RegionRequirement &req);
      
      static void createImageFieldPointer(LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                          Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]);
      
      static int numTreeLevels(ImageSize imageSize);
      
      static int subtreeHeight(ImageSize imageSize);
      
      static FutureMap launchTreeReduction(ImageSize imageSize, int treeLevel, int maxTreeLevel);
      
      static void launchPipelineReduction(ScreenSpaceArguments args);
      
      static int subdomainToCompositeIndex(SimulationBoundsCoordinate *bounds, int scale);
      
      
      ImageSize mImageSize;
      Context mContext;
      Runtime *mRuntime;
      LogicalRegion mSourceImage;
      LogicalRegion mScratchImage;
      Domain mSourceImageDomain;
      Domain mScratchImageDomain;
      Domain mDepthDomain;
      Domain mCompositePipelineDomain;
      Domain mDisplayDomain;
      Domain mSourceFragmentDomain;
      Domain mScratchFragmentDomain;
      LogicalPartition mDepthPartition;
      LogicalPartition mSourceFragmentPartition;
      LogicalPartition mScratchFragmentPartition;
      TaskID mInitialTaskID;
      TaskID mAccessorTaskID;
      TaskID mCompositeTaskID;
      TaskID mDisplayTaskID;
      GLenum mDepthFunction;
      GLenum mBlendFunctionSource;
      GLenum mBlendFunctionDestination;
      int mAccessorFunctorID;
      int mLocalCopyOfNodeID;
      
    public:
      static const int fieldsPerSimulationBounds = 2 * IMAGE_REDUCTION_DIMENSIONS;
      static int* mNodeID;
      static SimulationBoundsCoordinate *mMySimulationBounds;
      static SimulationBoundsCoordinate *mSimulationBounds;
      static int mNumSimulationBounds[IMAGE_REDUCTION_DIMENSIONS];
      static SimulationBoundsCoordinate mXMax, mXMin, mYMax, mYMin, mZMax, mZMin;
      static AccessorProjectionFunctor **mAccessorFunctor;
      static int mNumFunctors;
    };
    
  }
}

#endif /* image_reduction_h */
