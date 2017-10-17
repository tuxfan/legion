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


#ifndef legion_visualization_h
#define legion_visualization_h

#include "legion.h"

namespace Legion {
  namespace Profile {
    
    static const int image_region_dimensions = 1;//(width x height) x layerID
    
    typedef struct {
      int numNodes;
      
      Point<image_region_dimensions> origin() const{ return Point<image_region_dimensions>::ZEROES(); }
      
      Point<image_region_dimensions> upperBound() const{
        Point<image_region_dimensions> result;
        result[0] = numNodes;
        return result;
      }
      
      // launch by depth plane, each depth point is one image
      Point<image_region_dimensions> layerSize() const{
        Point<image_region_dimensions> result;
        result[0] = 1;
        return result;
      }
      Point<image_region_dimensions> numLayers() const{
        Point<image_region_dimensions> result;
        result[0] = numNodes;
        return result;
      }
      
      // launch by composite fragment,
      Point<image_region_dimensions> fragmentSize() const{
        Point<image_region_dimensions> result;
        result[0] = 1;
        return result;
      }
      Point<image_region_dimensions> numFragments() const{
        Point<image_region_dimensions> result;
        result[0] = numNodes;
        return result;
      }
      
      
      
    } ProfSize;
    
  }
}

#include "prof_reduction.h"


#endif /* legion_visualization_h */
