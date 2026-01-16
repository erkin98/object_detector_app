import datetime

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        if self._end is None:
             # if stop() hasn't been called, use current time for elapsed calculation
             # This allows checking FPS while running
             return (datetime.datetime.now() - self._start).total_seconds()
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        elapsed = self.elapsed()
        if elapsed == 0:
            return 0.0
        return self._numFrames / elapsed
