#include "image_segment.h"
#include "thread_sync.h"
#include "fps.h"

#include <thread>

static std::shared_ptr<cv::Mat> frame_to_process;
static std::shared_ptr<cv::Mat> frame_to_display;
static ThreadSync process_sync;
static ThreadSync display_sync;
static bool terminate = false;
static constexpr char win[] = "PhotoBooth";

void capture_thread() {
    // Request MJPEG, 1280x720, 30fps
    cv::VideoCapture camera("v4l2src ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink");

    while (!terminate) {
        // Keep 'em coming
        auto frame = std::make_shared<cv::Mat>();
        camera >> *frame;
        process_sync.produce([&] { frame_to_process = frame; });
    }

    // Make sure that process thread is unblocked
    process_sync.produce([] {});
}

void process_thread() {
    ImageSegment model;
    std::shared_ptr<cv::Mat> frame;

    while (!terminate) {
        process_sync.consume(
            [&] {
                // Check if there's a new frame available to process
                return terminate || frame != frame_to_process;
            },
            [&] {
                // Make sure it doesn't get garbage-collected
                if (!terminate && frame != frame_to_process) {
                    frame = frame_to_process;
                }
            },
            [&] {
                // Process the frame
                if (!terminate && frame) {
                    model.process_frame(frame);
                    display_sync.produce([&] { frame_to_display = frame; });
                }
            });
    }

    // Make sure that display thread is unblocked
    display_sync.produce([] {});
}

void display_thread() {
    Fps fps;
    std::shared_ptr<cv::Mat> frame;
    while (!terminate) {
        display_sync.consume(
            [&] {
                // Check if there's a new frame available to display
                return terminate || frame != frame_to_display;
            },
            [&] {
                // Make sure it doesn't get garbage-collected
                if (!terminate && frame != frame_to_display) {
                    frame = frame_to_display;
                }
            },
            [&] {
                // Display the frame
                if (!terminate && frame) {
                    cv::imshow(win, *frame);
                    fps.tick(60);
                }
            });
    }
}

int main(int argc __attribute__((unused)), char** argv __attribute__((unused)))
{
    // Create the window in advance
    cv::namedWindow(win);

    std::thread capture([] { capture_thread(); });
    std::thread process([] { process_thread(); });
    std::thread display([] { display_thread(); });

    // Wait for the Escape key
    while ((cv::waitKey(15) & 0xFF) != 27);

    terminate = true;
    capture.join();
    process.join();
    display.join();

    return 0;
}
