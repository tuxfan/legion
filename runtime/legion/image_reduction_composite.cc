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

inline void ImageReductionComposite::compositePixelsNever(PixelField *r0,
                                                          PixelField *g0,
                                                          PixelField *b0,
                                                          PixelField *a0,
                                                          PixelField *z0,
                                                          PixelField *userdata0,
                                                          PixelField *r1,
                                                          PixelField *g1,
                                                          PixelField *b1,
                                                          PixelField *a1,
                                                          PixelField *z1,
                                                          PixelField *userdata1,
                                                          PixelField *rOut,
                                                          PixelField *gOut,
                                                          PixelField *bOut,
                                                          PixelField *aOut,
                                                          PixelField *zOut,
                                                          PixelField *userdataOut,
                                                          int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
        r0++; g0++; b0++; a0++; z0++; userdata0++;
    }
    
}


inline void ImageReductionComposite::compositePixelsLess(PixelField *r0,
                                                         PixelField *g0,
                                                         PixelField *b0,
                                                         PixelField *a0,
                                                         PixelField *z0,
                                                         PixelField *userdata0,
                                                         PixelField *r1,
                                                         PixelField *g1,
                                                         PixelField *b1,
                                                         PixelField *a1,
                                                         PixelField *z1,
                                                         PixelField *userdata1,
                                                         PixelField *rOut,
                                                         PixelField *gOut,
                                                         PixelField *bOut,
                                                         PixelField *aOut,
                                                         PixelField *zOut,
                                                         PixelField *userdataOut,
                                                         int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 < *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}


inline void ImageReductionComposite::compositePixelsEqual(PixelField *r0,
                                                          PixelField *g0,
                                                          PixelField *b0,
                                                          PixelField *a0,
                                                          PixelField *z0,
                                                          PixelField *userdata0,
                                                          PixelField *r1,
                                                          PixelField *g1,
                                                          PixelField *b1,
                                                          PixelField *a1,
                                                          PixelField *z1,
                                                          PixelField *userdata1,
                                                          PixelField *rOut,
                                                          PixelField *gOut,
                                                          PixelField *bOut,
                                                          PixelField *aOut,
                                                          PixelField *zOut,
                                                          PixelField *userdataOut,
                                                          int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 == *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}


inline void ImageReductionComposite::compositePixelsLEqual(PixelField *r0,
                                                           PixelField *g0,
                                                           PixelField *b0,
                                                           PixelField *a0,
                                                           PixelField *z0,
                                                           PixelField *userdata0,
                                                           PixelField *r1,
                                                           PixelField *g1,
                                                           PixelField *b1,
                                                           PixelField *a1,
                                                           PixelField *z1,
                                                           PixelField *userdata1,
                                                           PixelField *rOut,
                                                           PixelField *gOut,
                                                           PixelField *bOut,
                                                           PixelField *aOut,
                                                           PixelField *zOut,
                                                           PixelField *userdataOut,
                                                           int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 <= *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}


inline void ImageReductionComposite::compositePixelsGreater(PixelField *r0,
                                                            PixelField *g0,
                                                            PixelField *b0,
                                                            PixelField *a0,
                                                            PixelField *z0,
                                                            PixelField *userdata0,
                                                            PixelField *r1,
                                                            PixelField *g1,
                                                            PixelField *b1,
                                                            PixelField *a1,
                                                            PixelField *z1,
                                                            PixelField *userdata1,
                                                            PixelField *rOut,
                                                            PixelField *gOut,
                                                            PixelField *bOut,
                                                            PixelField *aOut,
                                                            PixelField *zOut,
                                                            PixelField *userdataOut,
                                                            int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 > *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}

inline void ImageReductionComposite::compositePixelsNotEqual(PixelField *r0,
                                                             PixelField *g0,
                                                             PixelField *b0,
                                                             PixelField *a0,
                                                             PixelField *z0,
                                                             PixelField *userdata0,
                                                             PixelField *r1,
                                                             PixelField *g1,
                                                             PixelField *b1,
                                                             PixelField *a1,
                                                             PixelField *z1,
                                                             PixelField *userdata1,
                                                             PixelField *rOut,
                                                             PixelField *gOut,
                                                             PixelField *bOut,
                                                             PixelField *aOut,
                                                             PixelField *zOut,
                                                             PixelField *userdataOut,
                                                             int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 != *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}


inline void ImageReductionComposite::compositePixelsGEqual(PixelField *r0,
                                                           PixelField *g0,
                                                           PixelField *b0,
                                                           PixelField *a0,
                                                           PixelField *z0,
                                                           PixelField *userdata0,
                                                           PixelField *r1,
                                                           PixelField *g1,
                                                           PixelField *b1,
                                                           PixelField *a1,
                                                           PixelField *z1,
                                                           PixelField *userdata1,
                                                           PixelField *rOut,
                                                           PixelField *gOut,
                                                           PixelField *bOut,
                                                           PixelField *aOut,
                                                           PixelField *zOut,
                                                           PixelField *userdataOut,
                                                           int numPixels) {
    
    for(int i = 0; i < numPixels; ++i) {
        if(*z0 >= *z1) {
            *rOut++ = *r0++; *gOut++ = *g0++; *bOut++ = *b0++; *aOut++ = *a0++; *zOut++ = *z0++; *userdataOut++ = *userdata0++;
            r1++; g1++; b1++; a1++; z1++; userdata1++;
        } else {
            *rOut++ = *r1++; *gOut++ = *g1++; *bOut++ = *b1++; *aOut++ = *a1++; *zOut++ = *z1++; *userdataOut++ = *userdata1++;
            r0++; g0++; b0++; a0++; z0++; userdata0++;
        }
    }
    
}

inline void ImageReductionComposite::compositePixelsAlways(PixelField *r0,
                                                    PixelField *g0,
                                                    PixelField *b0,
                                                    PixelField *a0,
                                                    PixelField *z0,
                                                    PixelField *userdata0,
                                                    PixelField *r1,
                                                    PixelField *g1,
                                                    PixelField *b1,
                                                    PixelField *a1,
                                                    PixelField *z1,
                                                    PixelField *userdata1,
                                                    PixelField *rOut,
                                                    PixelField *gOut,
                                                    PixelField *bOut,
                                                    PixelField *aOut,
                                                    PixelField *zOut,
                                                    PixelField *userdataOut,
                                                    int numPixels) {
    
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


