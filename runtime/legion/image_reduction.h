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

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

namespace Legion {
    namespace Visualization {
        
        class ImageReduction {
            
        private:
            
            typedef struct {
                ImageSize imageSize;
                int totalSize;
                bool isAssociative;
                int compositeTaskID;
                GLenum depthFunction;
                GLenum blendFunctionSource;
                GLenum blendFunctionDestination;
                LogicalRegion image;
            } ScreenSpaceArguments;
            
            typedef struct {
                ImageSize imageSize;
                int x;
                int y;
                int layer0;
                int layer1;
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
            FutureMap launch_task_by_depth(unsigned taskID, void *args = NULL, int argLen = 0);
            /**
             * Perform a tree reduction using an associative commutative operator.
             * Be sure to call either set_blend_func or set_depth_func first.
             */
            FutureMap reduce_associative_commutative();
            /**
             * Perform a tree reduction using an associative noncommutative operator.
             * Be sure to call either set_blend_func or set_depth_func first.
             *
             * @param ordering integer permutation that defines ordering by depth index
             */
            FutureMap reduce_associative_noncommutative(int ordering[]);
            /**
             * Perform a pipeline reduction using a nonassociative commutative operator.
             * Be sure to call either set_blend_func or set_depth_func first.
             */
            FutureMap reduce_nonassociative_commutative();
            /**
             * Perform a pipeline reduction using a nonassociative noncommutative operator.
             * Be sure to call either set_blend_func or set_depth_func first.
             *
             * @param ordering integer permutation that defines ordering by depth index
             */
            FutureMap reduce_nonassociative_noncommutative(int ordering[]);
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
             * @param origin origin of image fragment
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
                                                    Point<IMAGE_REDUCTION_DIMENSIONS> origin,
                                                    PixelField *&r,
                                                    PixelField *&g,
                                                    PixelField *&b,
                                                    PixelField *&a,
                                                    PixelField *&z,
                                                    PixelField *&userdata,
                                                    ByteOffset stride[IMAGE_REDUCTION_DIMENSIONS]);
            
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
            
            static int composite_task(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime *runtime);
            
            static void screen_space_task(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context ctx, Runtime *runtime);
            
            
        private:
            
            FieldSpace imageFields();
            void createImage();
            void partitionImageByDepth();
            void partitionImageByScreenSpace();
            
            char* screenSpaceArguments(int ordering[], bool isAssociative);
            FutureMap launchScreenSpaceTasks(int ordering[], bool isAssociative);
            
            FutureMap reduceAssociative(int permutation[]);
            FutureMap reduceNonassociative(int permutation[]);
            
            void addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, ArgumentMap &argMap, int layer0, int layer1);
            void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);
            int *defaultPermutation();
            void registerTasks();
            
            static void addImageFieldsToRequirement(RegionRequirement &req);
            
            static void createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc, int fieldID, PixelField *&field,
                                                Rect<IMAGE_REDUCTION_DIMENSIONS> imageBounds, PhysicalRegion region, ByteOffset offset[]);
            
            static int numTreeLevels(ImageSize imageSize);
            
            static int subtreeHeight(ImageSize imageSize);
            
            static void addSubregionRequirementToFragmentLauncher(TaskLauncher &launcher, DomainPoint origin, int layer,
                                                                  Context context, Runtime* runtime, LogicalPartition partition, LogicalRegion parent);
            
            static Future launchCompositeTask(DomainPoint origin, int taskNumber, ScreenSpaceArguments args, FutureSet futures, int layer0, int layer1,
                                              Context context, Runtime* runtime, LogicalPartition fragmentPartition);
            
            static LogicalPartition partitionScreenSpaceByFragment(DomainPoint origin, Runtime* runtime, Context context, LogicalRegion parent, ImageSize imageSize);

            
            static FutureSet launchTreeReduction(ScreenSpaceArguments args, int *ordering, int treeLevel, int maxTreeLevel, DomainPoint origin,
                                                 Context context, Runtime* runtime, LogicalPartition fragmentPartition);
            
            static void launchPipelineReduction(ScreenSpaceArguments args, int *ordering);
            
            static PhysicalRegion compositeTwoFragments(CompositeArguments args, PhysicalRegion region0, Point<IMAGE_REDUCTION_DIMENSIONS> origin0,
                                                        PhysicalRegion region1, Point<IMAGE_REDUCTION_DIMENSIONS> origin1);
            
            ImageSize mImageSize;
            Context mContext;
            Runtime *mRuntime;
            LogicalRegion mImage;
            Domain mImageDomain;
            Domain mDepthDomain;
            Domain mCompositePipelineDomain;
            Domain mDisplayDomain;
            LogicalPartition mDepthPartition;
            LogicalPartition mScreenSpacePartition;
            Domain mScreenSpaceDomain;
            int *mDefaultPermutation;
            TaskID mScreenSpaceTaskID;
            TaskID mCompositeTaskID;
            TaskID mDisplayTaskID;
            GLenum mDepthFunction;
            GLenum mBlendFunctionSource;
            GLenum mBlendFunctionDestination;
        };
        
    }
}

#endif /* image_reduction_h */
