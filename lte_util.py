class EventScheduler(object):
    """ The event scheduler class for LTE. The scheduler suppports three main methods:
        - getNextEvent() for getting the subframe of the very next event
        - addEvent(events) for adding a list containing subframes of several events
        - clear() for emptying the event_queue
    """
    def __init__(self, *args, **kwargs):
        self.event_queue = []
        super().__init__(*args, **kwargs)

    def getNextEvent(self):
        return self.event_queue.pop()

    def peekNextEvent(self):
        return self.event_queue[-1]

    def addEvent(self, events):
        # https://stackoverflow.com/questions/53023380/bisect-insort-complexity-not-as-expected
        if len(events) > 0:
            events = set(events)
            self.event_queue += list(events)
            self.event_queue = sorted(set(self.event_queue), reverse=True)

    def clear(self):
        self.event_queue = []
