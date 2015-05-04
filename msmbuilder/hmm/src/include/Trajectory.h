#ifndef MIXTAPE_TRAJECTORY_H
#define MIXTAPE_TRAJECTORY_H

namespace msmbuilder {

/**
 * Trajectory provides a convenient wrapping of the data from a Numpy array describing a sequence of observations.
 * This class references the data as a char* regardless of data type, but get() is templatized based on the data type.
 * This means you can create a Trajectory without worrying about the data type, but you do need to know the data type
 * when you access it.
 */
class Trajectory {
public:
    /**
     * Create a Trajectory object wrapping an existing array of data.
     *
     * @param data           the existing data to wrap
     * @param numFrames      the number of frames in the trajectory
     * @param numFeatures    the number of features in each frame
     * @param frameStride    the offset between successive frames, measured in bytes
     * @param featureStride  the offset between successive features, measured in bytes
     */
    Trajectory(char* data, int numFrames, int numFeatures, int frameStride, int featureStride) :
        data(data), numFrames(numFrames), numFeatures(numFeatures), frameStride(frameStride), featureStride(featureStride) {
    }
    Trajectory() {
    }
    /**
     * Get the number of frames in the Trajectory.
     */
    int frames() const {
        return numFrames;
    }
    /**
     * Get the number of features in the Trajectory.
     */
    int features() const {
        return numFeatures;
    }
    /**
     * Get the value of a particular feature in a particular frame.
     */
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

} // namespace msmbuilder

#endif