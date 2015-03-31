#ifndef MIXTAPE_TRAJECTORY_H
#define MIXTAPE_TRAJECTORY_H

namespace Mixtape {

class Trajectory {
public:
    Trajectory(char* data, int numFrames, int numFeatures, int frameStride, int featureStride) :
        data(data), numFrames(numFrames), numFeatures(numFeatures), frameStride(frameStride), featureStride(featureStride) {
    }
    Trajectory() {
    }
    int frames() const {
        return numFrames;
    }
    int features() const {
        return numFeatures;
    }
    template <class T>
    const T& get(int frame, int feature) const {
        T* ptr = (T*) data;
        return ptr[(frame*frameStride+feature*featureStride)/sizeof(T)];
    }
private:
    char* data;
    int numFrames, numFeatures;
    int frameStride, featureStride;
};

} // namespace Mixtape

#endif