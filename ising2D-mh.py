import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation


def ising_step(spin, beta, num_subsamp=1000):
    """Computes a step of the Metropolis-Hasting algorithm
    for the 2D Ising model.

    Parameters
    ----------
    spin
        2D matrix representing spin field
    beta
        1/kT, where T is the absolute temperature and k Boltzmann's constant
    num_subsamp
        Number of sub-samples to use in MH

    Returns
    -------
    spin
        2D spin field with one spin changed due to interactions

    """
    N = spin.shape[0]

    # Sub-sample a num_subsamp times
    for flip_pos in np.random.randint(0, N, (num_subsamp, 2)):
        il, jl = flip_pos

        dE = 0
        for d in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i, j = d
            dE += spin[(il + i) % N, (jl + j) % N]
        dE = 2 * spin[il, jl] * dE

        if dE <= 0:
            # Accept with probabilty 1
            spin[il, jl] *= -1
        elif np.exp(-beta * dE) >= np.random.rand():
            # Accept with probabilty exp(-beta * dE)
            spin[il, jl] *= -1
    return spin


def ising_sample_gen(beta_targ, spin_shape=(20, 20), burnin=200, num_samples=2000, num_startvalues=1):
    """Generate `num_samples` samples of a 2D Ising model.

    Parameters
    ----------
    beta_targ
        target value for beta
    spin_shape
        Size of spin grid (width, height)
    burnin
        Number of samples to throw away for burnin
    num_samples
        How many samples to use
    num_startvalues
        Number of startvalues for sample generation

    Returns
    -------
    samples
        Samples for a 2D Ising model
    """
    samples = np.zeros((num_samples + burnin, spin_shape[0], spin_shape[1]), dtype=np.int8)
    # Initialization (generates random 2D spin fields at samples[0], ..., samples[num_startvalues])
    samples[:num_startvalues, :, :] = np.random.choice([-1, 1], (num_startvalues,) + spin_shape)

    # Generate discrete temperature points
    betas = np.linspace(start=1, stop=beta_targ, num=num_samples + burnin)

    for i in range(num_startvalues, num_samples + burnin):
        samples[i, :, :] = ising_step(samples[i - num_startvalues, :, :], betas[i])

    # Avoiding the first few samples because they are likely not very good
    return samples[burnin:, :]


# -----> SCRIPT STARTS HERE <-----

# Define target value for beta (critical temp)
beta = np.log(1 + np.sqrt(2)) / 2

fig = plt.figure()

# Generate samples for 2D Ising model
samples = ising_sample_gen(beta, spin_shape=(200, 200), num_startvalues=1, burnin=50, num_samples=1500)

# Init matplotlib plot
im = plt.imshow(samples[0, :, :], cmap="viridis")

# Matplotlib animation helper functions
def init():
    im.set_data(samples[0, :, :])
    return (im,)

def animate(i):
    im.set_array(samples[i, :, :])
    return (im,)

# Generate animation
anim = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=25, blit=True)
# Save animation to mp4
anim.save("ising2d.mp4", fps=30, extra_args=["-vcodec", "libx264"])
