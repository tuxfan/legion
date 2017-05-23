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


#ifndef ImageReductionComposite_h
#define ImageReductionComposite_h

#include "legion.h"
#include "legion_visualization.h"

#include "image_reduction.h"

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <stdio.h>


using namespace Legion::Visualization;

class ImageReductionComposite {
public:
    typedef void(CompositeFunction)
    (PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*,
    PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*,
    PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, PixelField*, int);

    static CompositeFunction* compositeFunctionPointer(GLenum depthFunction, GLenum blendFunctionSource, GLenum blendFunctionDestination);

    
    static CompositeFunction compositePixelsNever;
    static CompositeFunction compositePixelsLess;
    static CompositeFunction compositePixelsEqual;
    static CompositeFunction compositePixelsLEqual;
    static CompositeFunction compositePixelsGreater;
    static CompositeFunction compositePixelsNotEqual;
    static CompositeFunction compositePixelsGEqual;
    static CompositeFunction compositePixelsAlways;

private:
};



#endif /* ImageReductionComposite_h */
