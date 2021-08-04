import collections


DoneTransition = collections.namedtuple("Transition",
                                        ("observation",
                                         "action",
                                         "reward",
                                         "next_observation",
                                         "done"))
