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


#ifndef UsecTimer_h
#define UsecTimer_h

#include <chrono>

using namespace std;
//using std::chrono::high_resolution_clock;

class UsecTimer {
public:
    UsecTimer(string description){
        mDescription = description;
        mCumulativeElapsedSeconds = 0.0;
        mNumSamples = 0;
        mStarted = false;
    }
    ~UsecTimer(){}
    void start(){
        //mStart = high_resolution_clock::now();
        mStarted = true;
    }
    void stop(){
        if(mStarted) {
            //std::chrono::time_point<std::chrono::high_resolution_clock> end = high_resolution_clock::now();
            //double elapsedSeconds = ((end - mStart).count()) * high_resolution_clock::period::num /
            //static_cast<double>(high_resolution_clock::period::den);
            //mCumulativeElapsedSeconds += elapsedSeconds;
            mNumSamples++;
            mStarted = false;
        }
    }
    string to_string(){
        double meanSampleElapsedSeconds = 0;
        if(mNumSamples > 0) {
            meanSampleElapsedSeconds = mCumulativeElapsedSeconds / mNumSamples;
        }
        double sToUs = 1000000.0;
        return mDescription
        + " " + std::to_string(mCumulativeElapsedSeconds) + " sec"
        + " " + std::to_string(mCumulativeElapsedSeconds * sToUs)
        + " usec = " + std::to_string(meanSampleElapsedSeconds * sToUs)
        + " usec * " + std::to_string(mNumSamples)
        + (mNumSamples == 1 ? " sample" : " samples");
    }
    
private:
    bool mStarted;
    //high_resolution_clock::time_point mStart;
    string mDescription;
    double mCumulativeElapsedSeconds;
    int mNumSamples;
};


#endif /* UsecTimer_h */
