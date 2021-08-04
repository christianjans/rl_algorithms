import collections


Transition = collections.namedtuple("Transition",
                                    ("observation",
                                     "action",
                                     "reward",
                                     "next_observation"))


DoneTransition = collections.namedtuple("DoneTransition",
                                        ("observation",
                                         "action",
                                         "reward",
                                         "next_observation",
                                         "done"))
