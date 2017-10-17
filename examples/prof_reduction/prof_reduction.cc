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

#include "usec_timer.h"


using namespace Legion;
using namespace Legion::Profile;


enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  RENDER_TASK_ID,
};



static void simulateTimeStep(int t) {
  // tbd
}



static void paintRegion(ProfSize profSize,
                        ProfReduction::ProfileField *r,
                        ProfReduction::ProfileField *g,
                        ProfReduction::ProfileField *b,
                        ProfReduction::ProfileField *a,
                        ProfReduction::ProfileField *z,
                        ProfReduction::ProfileField *userdata,
                        ProfReduction::Stride stride,
                        int layer) {
  
  ProfReduction::ProfileField zValue = layer;
  for(int row = 0; row < profSize.height; ++row) {
    for(int column = 0; column < profSize.width; ++column) {
      *r = layer;
      *g = layer;
      *b = layer;
      *a = layer;
      *z = layer;
      *userdata = layer;
      r += stride[ProfReduction::FID_FIELD_R][0];
      g += stride[ProfReduction::FID_FIELD_G][0];
      b += stride[ProfReduction::FID_FIELD_B][0];
      a += stride[ProfReduction::FID_FIELD_A][0];
      z += stride[ProfReduction::FID_FIELD_Z][0];
      userdata += stride[ProfReduction::FID_FIELD_USERDATA][0];
      zValue = (zValue + 1);
      zValue = (zValue >= profSize.numImageLayers) ? 0 : zValue;
    }
  }
}

void render_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {
  
  UsecTimer render(Legion::Profile::ProfReduction::describe_task(task) + ":");
  render.start();
  PhysicalRegion image = regions[0];
  ProfSize profSize = ((ProfSize *)task->args)[0];
  
  ProfReduction::ProfileField *r, *g, *b, *a, *z, *userdata;
  ProfReduction::Stride stride;
  int layer = task->get_unique_id() % profSize.numImageLayers;
  ProfReduction::create_image_field_pointers(profSize, image, r, g, b, a, z, userdata, stride, runtime, ctx);
  paintRegion(profSize, r, g, b, a, z, userdata, stride, layer);
  render.stop();
  cout << render.to_string() << endl;
}



void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
  

#ifdef IMAGE_SIZE
  ProfSize profSize = (ProfSize){ IMAGE_SIZE };
  
#else
  const int width = 3840;
  const int height = 2160;
  const int numSimulationTasks = 4;
  const int numFragmentsPerLayer = 8;
  
  ProfSize profSize = (ProfSize){ width, height, numSimulationTasks, numFragmentsPerLayer };
#endif
  
  std::cout << "ProfSize (" << profSize.width << "," << profSize.height
  << ") x " << profSize.numImageLayers << " layers " << profSize.numFragmentsPerLayer << " frags/layer" << std::endl;
  
  ProfReduction profReduction(profSize, ctx, runtime);
  {
    UsecTimer overall("overall time:");
    overall.start();
    UsecTimer frame("frame time:");
    UsecTimer reduce("reduce time:");
    Future displayFuture;
    
    const int numTimeSteps = 5;
    
    for(int t = 0; t < numTimeSteps; ++t) {
      
      frame.start();
      simulateTimeStep(t);
      FutureMap renderFutures = profReduction.launch_task_by_nodeID(RENDER_TASK_ID);
      renderFutures.wait_all_results();
      
      reduce.start();
      FutureMap reduceFutures = profReduction.reduce_associative_commutative();
      reduceFutures.wait_all_results();
      reduce.stop();
      
      frame.stop();
    }
    
    overall.stop();
    
    std::cout << reduce.to_string() << std::endl;
    std::cout << frame.to_string() << std::endl;
    std::cout << overall.to_string() << std::endl;
    
  }
}





int main(const int argc, char *argv[]) {
  
  Legion::Profile::ProfReduction::initialize();
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
                                                         Processor::LOC_PROC, true/*single*/, false/*index*/,
                                                         AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "topLevelTask");
    
  return HighLevelRuntime::start(argc, argv);
}
