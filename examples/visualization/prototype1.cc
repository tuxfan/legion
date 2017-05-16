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


#include <iostream>

#include "legion.h"
#include "legion_visualization.h"

#include "RenderSpace.h"
#include "UsecTimer.h"

#ifndef TIME_PER_FRAME
#define TIME_PER_FRAME 1
#endif

#ifndef TIME_OVERALL
#define TIME_OVERALL 1
#endif

#ifndef TREE_REDUCTION
#define TREE_REDUCTION 1
#endif

#define ACCESSOR_TYPE AccessorType::Generic

using namespace std;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace Legion::Visualization;


enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    RENDER_TASK_ID,
};



static void simulateTimeStep(int t) {
    // tbd
}



static void paintRegion(ImageSize imageSize,
                        PixelField *r,
                        PixelField *g,
                        PixelField *b,
                        PixelField *a,
                        PixelField *z,
                        PixelField *userdata,
                        ByteOffset stride[DIMENSIONS],
                        int taskID) {
    
    PixelField zValue = taskID % imageSize.depth;
    
    for(int row = 0; row < imageSize.height; ++row) {
#pragma unroll
        for(int column = 0; column < imageSize.width; ++column) {
            *r = taskID;
            *g = taskID;
            *b = taskID;
            *a = taskID;
            *z = zValue;
            *userdata = taskID;
            r += stride[0]; g += stride[0]; b += stride[0]; a += stride[0]; z += stride[0]; userdata += stride[0];
            zValue = (zValue + 1);
            zValue = (zValue >= imageSize.depth) ? 0 : zValue;
        }
    }
}

void render_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {
    
    UsecTimer render(RenderSpace::describeTask(task) + ":");
    render.start();
    PhysicalRegion image = regions[0];
    ImageSize imageSize = ((ImageSize *)task->args)[0];
    
    PixelField *r, *g, *b, *a, *z, *userdata;
    ByteOffset stride[DIMENSIONS];
    int layer = task->get_unique_id() % imageSize.depth;
    RenderSpace::createImageFieldPointers(imageSize, image, layer, r, g, b, a, z, userdata, stride);
    paintRegion(imageSize, r, g, b, a, z, userdata, stride, task->get_unique_id());
    render.stop();
    cout << render.to_string() << endl;
}



void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
    
    runtime->attach_name(TOP_LEVEL_TASK_ID, "top_level_task");
    runtime->attach_name(RENDER_TASK_ID, "render_task");
    
    const int numSimulationTasks = 4;
    const int numTimeSteps = 5;
    
#if 0
    const int width = 3840;
    const int height = 2160;
#elif 1
    const int width = 2048;
    const int height = 1024;
#elif 0
    const int width = 512;
    const int height = 128;
#else
    const int width = 16;
    const int height = 8;
#endif

#ifdef NUM_FRAGMENTS_PER_LAYER
    const int numFragmentsPerLayer = NUM_FRAGMENTS_PER_LAYER;
#else
    const int numFragmentsPerLayer = numSimulationTasks * 8;
#endif
    
    ImageSize imageSize = (ImageSize){ width, height, numSimulationTasks, numFragmentsPerLayer };
    RenderSpace *renderSpace = new RenderSpace(imageSize, ctx, runtime);
    
    {
        
        UsecTimer overall("overall time:");
        overall.start();
        UsecTimer frame("frame time:");
        UsecTimer reduce("reduce time:");
        Future displayFuture;
        
        for(int t = 0; t < numTimeSteps; ++t) {
            frame.start();
            simulateTimeStep(t);
            FutureMap renderFutures = renderSpace->launchTaskByDepth(RENDER_TASK_ID);
            reduce.start();
            
#if TREE_REDUCTION
            Futures reduceFutures = renderSpace->reduceAssociativeCommutative();
#else
            Futures reduceFutures = renderSpace->reduceNonassociativeCommutative();
#endif
            
            
#if TIME_PER_FRAME
            std::cout << "waiting for reduction" << std::endl;
            reduceFutures[0].wait();
#endif
            reduce.stop();
            
            displayFuture = renderSpace->display(t);
            
#if TIME_PER_FRAME
            std::cout << "waiting for display" << std::endl;
            displayFuture.wait();
#endif
            frame.stop();
        }
        
#if TIME_OVERALL
        displayFuture.wait();
#endif
        
        overall.stop();
        
#if TIME_PER_FRAME
        cout << frame.to_string() << endl;
        cout << reduce.to_string() << endl;
#endif
#if TIME_OVERALL
        cout << overall.to_string() << endl;
#endif
    }
    
    delete renderSpace;
}





int main(const int argc, char *argv[]) {
    
    HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
                                                           Processor::LOC_PROC, true/*single*/, false/*index*/);
    HighLevelRuntime::register_legion_task<render_task>(RENDER_TASK_ID,
                                                        Processor::LOC_PROC, false/*single*/, true/*index*/);
    
    RenderSpace::registerTasks();
    
    return HighLevelRuntime::start(argc, argv);
}
