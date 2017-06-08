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

#include "image_reduction_composite.h"

namespace Legion {
  namespace Visualization {
    
        
    inline void ImageReductionComposite::compositePixelsNever(ImageReduction::PixelField *r0,
                                                              ImageReduction::PixelField *g0,
                                                              ImageReduction::PixelField *b0,
                                                              ImageReduction::PixelField *a0,
                                                              ImageReduction::PixelField *z0,
                                                              ImageReduction::PixelField *userdata0,
                                                              ImageReduction::PixelField *r1,
                                                              ImageReduction::PixelField *g1,
                                                              ImageReduction::PixelField *b1,
                                                              ImageReduction::PixelField *a1,
                                                              ImageReduction::PixelField *z1,
                                                              ImageReduction::PixelField *userdata1,
                                                              ImageReduction::PixelField *rOut,
                                                              ImageReduction::PixelField *gOut,
                                                              ImageReduction::PixelField *bOut,
                                                              ImageReduction::PixelField *aOut,
                                                              ImageReduction::PixelField *zOut,
                                                              ImageReduction::PixelField *userdataOut,
                                                              int numPixels,
                                                              Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    
    inline void ImageReductionComposite::compositePixelsLess(ImageReduction::PixelField *r0,
                                                             ImageReduction::PixelField *g0,
                                                             ImageReduction::PixelField *b0,
                                                             ImageReduction::PixelField *a0,
                                                             ImageReduction::PixelField *z0,
                                                             ImageReduction::PixelField *userdata0,
                                                             ImageReduction::PixelField *r1,
                                                             ImageReduction::PixelField *g1,
                                                             ImageReduction::PixelField *b1,
                                                             ImageReduction::PixelField *a1,
                                                             ImageReduction::PixelField *z1,
                                                             ImageReduction::PixelField *userdata1,
                                                             ImageReduction::PixelField *rOut,
                                                             ImageReduction::PixelField *gOut,
                                                             ImageReduction::PixelField *bOut,
                                                             ImageReduction::PixelField *aOut,
                                                             ImageReduction::PixelField *zOut,
                                                             ImageReduction::PixelField *userdataOut,
                                                             int numPixels,
                                                             Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]){
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 < *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    
    inline void ImageReductionComposite::compositePixelsEqual(ImageReduction::PixelField *r0,
                                                              ImageReduction::PixelField *g0,
                                                              ImageReduction::PixelField *b0,
                                                              ImageReduction::PixelField *a0,
                                                              ImageReduction::PixelField *z0,
                                                              ImageReduction::PixelField *userdata0,
                                                              ImageReduction::PixelField *r1,
                                                              ImageReduction::PixelField *g1,
                                                              ImageReduction::PixelField *b1,
                                                              ImageReduction::PixelField *a1,
                                                              ImageReduction::PixelField *z1,
                                                              ImageReduction::PixelField *userdata1,
                                                              ImageReduction::PixelField *rOut,
                                                              ImageReduction::PixelField *gOut,
                                                              ImageReduction::PixelField *bOut,
                                                              ImageReduction::PixelField *aOut,
                                                              ImageReduction::PixelField *zOut,
                                                              ImageReduction::PixelField *userdataOut,
                                                              int numPixels,
                                                              Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 == *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    
    inline void ImageReductionComposite::compositePixelsLEqual(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 <= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    
    inline void ImageReductionComposite::compositePixelsGreater(ImageReduction::PixelField *r0,
                                                                ImageReduction::PixelField *g0,
                                                                ImageReduction::PixelField *b0,
                                                                ImageReduction::PixelField *a0,
                                                                ImageReduction::PixelField *z0,
                                                                ImageReduction::PixelField *userdata0,
                                                                ImageReduction::PixelField *r1,
                                                                ImageReduction::PixelField *g1,
                                                                ImageReduction::PixelField *b1,
                                                                ImageReduction::PixelField *a1,
                                                                ImageReduction::PixelField *z1,
                                                                ImageReduction::PixelField *userdata1,
                                                                ImageReduction::PixelField *rOut,
                                                                ImageReduction::PixelField *gOut,
                                                                ImageReduction::PixelField *bOut,
                                                                ImageReduction::PixelField *aOut,
                                                                ImageReduction::PixelField *zOut,
                                                                ImageReduction::PixelField *userdataOut,
                                                                int numPixels,
                                                                Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 > *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    inline void ImageReductionComposite::compositePixelsNotEqual(ImageReduction::PixelField *r0,
                                                                 ImageReduction::PixelField *g0,
                                                                 ImageReduction::PixelField *b0,
                                                                 ImageReduction::PixelField *a0,
                                                                 ImageReduction::PixelField *z0,
                                                                 ImageReduction::PixelField *userdata0,
                                                                 ImageReduction::PixelField *r1,
                                                                 ImageReduction::PixelField *g1,
                                                                 ImageReduction::PixelField *b1,
                                                                 ImageReduction::PixelField *a1,
                                                                 ImageReduction::PixelField *z1,
                                                                 ImageReduction::PixelField *userdata1,
                                                                 ImageReduction::PixelField *rOut,
                                                                 ImageReduction::PixelField *gOut,
                                                                 ImageReduction::PixelField *bOut,
                                                                 ImageReduction::PixelField *aOut,
                                                                 ImageReduction::PixelField *zOut,
                                                                 ImageReduction::PixelField *userdataOut,
                                                                 int numPixels,
                                                                 Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 != *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    
    inline void ImageReductionComposite::compositePixelsGEqual(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 >= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
    }
    
    inline void ImageReductionComposite::compositePixelsAlways(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::ByteOffset stride[ImageReduction::numPixelFields][image_region_dimensions]) {
      
      // no change */
    }
    
    
    
    
    ImageReductionComposite::CompositeFunction* ImageReductionComposite::compositeFunctionPointer(GLenum depthFunction, GLenum blendFunctionSource, GLenum blendFunctionDestination) {
      if(depthFunction != 0) {
        switch(depthFunction) {
          case GL_NEVER: return compositePixelsNever;
          case GL_LESS: return compositePixelsLess;
          case GL_EQUAL: return compositePixelsEqual;
          case GL_LEQUAL: return compositePixelsLEqual;
          case GL_GREATER: return compositePixelsGreater;
          case GL_NOTEQUAL: return compositePixelsNotEqual;
          case GL_GEQUAL: return compositePixelsGEqual;
          case GL_ALWAYS: return compositePixelsAlways;
            
        }
      } else if(blendFunctionSource != 0 && blendFunctionDestination != 0) {
        assert(false);
      }
      return NULL;
    }
    
  }
}

