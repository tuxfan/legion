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
        const int DIMENSIONS = 3;
        
        typedef struct {
            int width;
            int height;
            int depth;
            int numFragmentsPerLayer;
            
            int pixelsPerPlane() const{ return width * height; }
            
            Point<DIMENSIONS> origin() const{ return Point<DIMENSIONS>::ZEROES(); }
            Point<DIMENSIONS> upperBound() const{
                Point<DIMENSIONS> result;
                result.x[0] = width;
                result.x[1] = height;
                result.x[2] = depth;
                return result;
            }
            
            // launch by depth plane, each depth point is one image
            Point<DIMENSIONS> layerSize() const{
                Point<DIMENSIONS> result;
                result.x[0] = width;
                result.x[1] = height;
                result.x[2] = 1;
                return result;
            }
            Point<DIMENSIONS> numLayers() const{
                Point<DIMENSIONS> result;
                result.x[0] = 1;
                result.x[1] = 1;
                result.x[2] = depth;
                return result;
            }
            
            // launch by composite fragment,
            Point<DIMENSIONS> fragmentSize() const{
                Point<DIMENSIONS> result;
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
            Point<DIMENSIONS> numFragments() const{
                Point<DIMENSIONS> result;
                Point<DIMENSIONS> size = fragmentSize();
                result.x[0] = width / size.x[0];
                result.x[1] = height / size.x[1];
                result.x[2] = depth;
                return result;
            }
                        
            Point<DIMENSIONS> incrementFragment(Point<DIMENSIONS> point) const {
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
                Point<DIMENSIONS> size = fragmentSize();
                int result = 1;
                for(int i = 0; i < DIMENSIONS; ++i) {
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

#endif /* legion_visualization_h */
