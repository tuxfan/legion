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

using namespace LegionRuntime::Arrays;

namespace Legion {
    namespace Visualization {
        
        const int NUM_FRAGMENTS_PER_COMPOSITE_TASK = 2;
        const int IMAGE_REDUCTION_DIMENSIONS = 3;
        
        typedef struct {
            int width;
            int height;
            int depth;
            int numFragmentsPerLayer;
            
            int pixelsPerLayer() const{ return width * height; }
            
            Point<IMAGE_REDUCTION_DIMENSIONS> origin() const{ return Point<IMAGE_REDUCTION_DIMENSIONS>::ZEROES(); }
            Point<IMAGE_REDUCTION_DIMENSIONS> upperBound() const{
                Point<IMAGE_REDUCTION_DIMENSIONS> result;
                result.x[0] = width;
                result.x[1] = height;
                result.x[2] = depth;
                return result;
            }
            
            // launch by depth plane, each depth point is one image
            Point<IMAGE_REDUCTION_DIMENSIONS> layerSize() const{
                Point<IMAGE_REDUCTION_DIMENSIONS> result;
                result.x[0] = width;
                result.x[1] = height;
                result.x[2] = 1;
                return result;
            }
            Point<IMAGE_REDUCTION_DIMENSIONS> numLayers() const{
                Point<IMAGE_REDUCTION_DIMENSIONS> result;
                result.x[0] = 1;
                result.x[1] = 1;
                result.x[2] = depth;
                return result;
            }
            
            // launch by composite fragment,
            Point<IMAGE_REDUCTION_DIMENSIONS> fragmentSize() const{
                Point<IMAGE_REDUCTION_DIMENSIONS> result;
                if(numFragmentsPerLayer > height) {
                    assert(width % numFragmentsPerLayer == 0);
                    result.x[0] = width / numFragmentsPerLayer;
                    result.x[1] = 1;
                    result.x[2] = 1;
                } else {
                    result.x[0] = width;
                    assert(height % numFragmentsPerLayer == 0);
                    result.x[1] = height / numFragmentsPerLayer;
                    result.x[2] = 1;
                }
                return result;
            }
            Point<IMAGE_REDUCTION_DIMENSIONS> numFragments() const{
                Point<IMAGE_REDUCTION_DIMENSIONS> result;
                Point<IMAGE_REDUCTION_DIMENSIONS> size = fragmentSize();
                result.x[0] = width / size.x[0];
                result.x[1] = height / size.x[1];
                result.x[2] = depth;
                return result;
            }
                        
            Point<IMAGE_REDUCTION_DIMENSIONS> incrementFragment(Point<IMAGE_REDUCTION_DIMENSIONS> point) const {
                point.x[0] += 1;
                if(point.x[0] >= numFragments().x[0]) {
                    point.x[0] = 0;
                    point.x[1] += 1;
                    if(point.x[1] >= numFragments().x[1]) {
                        point.x[1] = 0;
                        point.x[2] += 1;
                        if(point.x[2] >= numFragments().x[2]) {
                            point.x[2] = 0;
                        }
                    }
                }
                return point;
            }
            
            int numPixelsPerFragment() const {
                Point<IMAGE_REDUCTION_DIMENSIONS> size = fragmentSize();
                int result = 1;
                for(int i = 0; i < IMAGE_REDUCTION_DIMENSIONS; ++i) {
                    result *= size.x[i];
                }
                return result;
            }
            
        } ImageSize;
        
        typedef float PixelField;
        
        enum FieldIDs {
            FID_FIELD_R,
            FID_FIELD_G,
            FID_FIELD_B,
            FID_FIELD_A,
            FID_FIELD_Z,
            FID_FIELD_USERDATA,
        };
        
    }
}

#include "image_reduction.h"


#endif /* legion_visualization_h */
