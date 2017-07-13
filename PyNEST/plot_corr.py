import glob
import matplotlib.pyplot as plt
import numpy as np
import re


def parse_data(data, twarmup):
    data = data[data[:, 1] > twarmup]
    uidx = np.unique(data[:, 0])
    spiketrains = []
    for idx in uidx:
        spiketrains.append(data[data[:, 0] == idx][:, 1])
    return spiketrains


def find_basenames(datadir):
    datafile_names = glob.glob(datadir)
    basenames = {}
    for fn in datafile_names:
        layer = re.search('_(L[0-9]+[EI])-', fn).group(1)
        basename = re.search('(.*)[0-9]+.gdf', fn).group(1)
        basenames[layer] = basename
    return basenames


def population_firing_rate(spiketrains, tmin, tmax, tbin):
    st = np.hstack(spiketrains)
    bins = np.arange(tmin, tmax, tbin)
    hist, bins = np.histogram(st, bins=bins)
    return bins[:-1], hist


def load_data(basename, threads):
    return np.vstack([np.loadtxt('{basename}{thread}.gdf'.format(basename=basename, thread=thread)) for thread in range(threads)])


def euclidean_distance(v0, v1):
    diff = v0 - v1
    return np.sqrt(np.dot(diff, diff))


params = {
    # 'datadir_templates': ['./data_ind{trial}/*.gdf', './data_poisson{trial}/*.gdf', './data_network{trial}/*.gdf'],  # './data_dc/*.gdf',
    # 'datadir_templates': ['./data_ind{trial}_exc_dense/*.gdf', './data_poisson{trial}_exc_dense/*.gdf', './data_network{trial}_exc_dense/*.gdf'],  # './data_dc/*.gdf',
    'datadir_template': './data_{stim}_{seed}_{conn}/spike*.gdf',  # './data_dc/*.gdf',
    'stims': ['pool', 'network'],
    'threads': 2,
    'seeds': [55, 56, 57],
    'conns': [0.3, 0.5, 0.8, 0.9],
    'twarmup': 500.,
    'tmin': 0.,
    'tmax': 2000.,
    'tbin': 1.,
}

fig = plt.figure()
ax_cov_dist = fig.add_subplot(111)
ax_cov_dist.set_ylabel(r'$||\mathsf{cov}(\cdot) - \mathsf{cov}(\mathsf{ind})||$')
ax_cov_dist.set_xlabel(r'Pool size $N$')
ax_cov_dist.set_xscale('log')
ax_cov_dist.set_yscale('log')
ax_cov_dist.set_xlim([2900., 1e4])

rate = []
cov = []
rate_dist = {}
cov_dist = {}
cov_dist_std = {}

# for i, dtm in enumerate(params['datadir_templates']):
for stim in params['stims']:
    cov_dist[stim] = []
    cov_dist_std[stim] = []
    for conn in params['conns']:
        cov_dist_trial = []
        for seed in params['seeds']:
            basenames = find_basenames(params['datadir_template'].format(stim=stim, seed=seed, conn=conn))
            popfr = []
            for layer in sorted(basenames):
                popfr.append(population_firing_rate(parse_data(load_data(basenames[layer], params['threads']), params['twarmup']), params['tmin'], params['tmax'], params['tbin'])[1])

            rate.append(np.mean(popfr, axis=0))
            cov.append(np.cov(popfr))
            # ax = axes[i].imshow(cov[j] - cov[0], interpolation='nearest', cmap='viridis', vmin=-1., vmax=1.)
            # print(dtm.format(seed=seed), euclidean_distance(rate[j].flatten(), rate[0].flatten()), euclidean_distance(cov[j].flatten(), cov[0].flatten()))
            cov_dist_trial.append(euclidean_distance(cov[-1].flatten(), cov[0].flatten()))

        cov_dist[stim].append(np.mean(cov_dist_trial))
        cov_dist_std[stim].append(np.std(cov_dist_trial))

    # plt.plot(2900. / np.array(params['conns']), cov_dist[stim], marker='o')
    ax_cov_dist.errorbar(2900. / np.array(params['conns']), cov_dist[stim], yerr=cov_dist_std[stim], marker='o', label=stim)
# plt.colorbar(ax, ax=axes)
# plt.show()
ax_cov_dist.legend(loc='best')
fig.savefig('cov_dist.pdf')
