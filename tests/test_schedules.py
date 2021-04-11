import albumentations as A
import numpy as np
import matplotlib.pyplot as plt


def test_schedules():
    sched = A.CosineWarmupAndDecaySchedule(num_steps=1000, start_p=0.1, maximum_p=0.95, final_p=0.4)
    x = np.arange(sched.num_steps + 1)
    y = [sched.step(xi) for xi in x]

    plt.figure()
    plt.grid(True)
    plt.plot(x, y)

    sched = A.CosineWarmupAndDecaySchedule(
        num_steps=1000, peak_fraction=0.75, start_p=0.2, maximum_p=0.85, final_p=0.1
    )
    x = np.arange(sched.num_steps + 1)
    y = [sched.step(xi) for xi in x]

    plt.plot(x, y)
    plt.show()
