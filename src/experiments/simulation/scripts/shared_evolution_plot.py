if __name__ == '__main__':
    from src.util import load_from, samples2res, transform_weights_from_log,transform_p_from_log
    from src.preprocessing import compute_network, get_sites
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_minimum_spanning_tree
    import itertools
    import numpy as np

    PATH = '/home/sebastian/git_repos/contact_zones/src/experiments/simulation/'
    TEST_ZONE_DIRECTORY = 'results/shared_evolution/2019-10-20_17-57/'
    PLOT_PATH = '/home/sebastian/git_repos/contact_zones/plots/'

    # Zone, ease and number of runs

    # Zone [3, 4, 6, 8]
    zones = [3, 4, 6, 8]
    zone = 3

    # Ease [0, 1, 2]
    eases = [0, 1, 2]
    ease = 0

    # Run [0]
    runs = [0]
    run = 0


    scenarios = [zones, eases, runs]
    scenarios = list(itertools.product(*scenarios))
    print(scenarios)

    for scenario in scenarios:

        zone, ease, run = scenario

        # Load the MCMC results
        sample_path = f'{PATH}{TEST_ZONE_DIRECTORY}shared_evolution_z{zone}_e{ease}_{run}.pkl'
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)

        zones = mcmc_res['zones']
        print(f'Number of zones: {len(zones)}')
        # print(type(mcmc_res['zones']))
        # print(len(mcmc_res['zones'][0][0]))

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH}data/sites_simulation.csv')
        network = compute_network(sites)

        # Plot posterior frequency
        fname = f'{PLOT_PATH}posterior_frequency_z{zone}_e{ease}_{run}.png'
        plot_posterior_frequency(mcmc_res, net=network, nz=0, burn_in=0.6, fname=fname)

        # Plot minimum spanning tree
        fname = f'{PLOT_PATH}minimum spanning tree_z{zone}_e{ease}_{run}.png'
        plot_minimum_spanning_tree(mcmc_res, network, z=1, ts_posterior_freq=0.8, fname=fname)

        # Plot trace of likelihood, recall and precision
        fname = f'{PLOT_PATH}trace_likelihood_z{zone}_e{ease}_{run}.png'
        plot_trace_lh(mcmc_res, burn_in=0.4, true_lh=True, fname=fname)

        fname = f'{PLOT_PATH}trace_recall_precision_z{zone}_e{ease}_{run}.png'
        plot_trace_recall_precision(mcmc_res, burn_in=0.4, fname=fname)

        # Plot zone size over time
        fname = f'{PLOT_PATH}zone_size_over_time_z{zone}_e{ease}_{run}.png'
        plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2, fname=fname)


    """

    TEST_ZONE_DIRECTORY = 'results/number_zones/2019-10-24_14-27/'
    

    scenarios = [1, 2, 3, 4, 5, 6, 7]
    scenarios = [1, 2, 3, 4] # fix for more than 4 zones

    for n_zones in scenarios:

        # Load the MCMC results
        sample_path = f'{PATH}{TEST_ZONE_DIRECTORY}number_zones_nz{n_zones}_0.pkl'
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)

        zones = mcmc_res['zones']
        print(f'Number of zones: {len(zones)}')
        # print(type(mcmc_res['zones']))
        # print(len(mcmc_res['zones'][0][0]))

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH}data/sites_simulation.csv')
        network = compute_network(sites)

        # Plot posterior frequency
        fname = f'{PLOT_PATH}posterior_frequency_nz{n_zones}.png'
        plot_posterior_frequency(mcmc_res, net=network, nz=-1, show_zone_bbox=True, burn_in=0.6, fname=fname)

        # Plot minimum spanning tree
        fname = f'{PLOT_PATH}minimum spanning tree_nz{n_zones}.png'
        plot_minimum_spanning_tree(mcmc_res, network, z=1, ts_posterior_freq=0.8, fname=fname)

        # Plot trace of likelihood, recall and precision
        fname = f'{PLOT_PATH}trace_likelihood_nz{n_zones}.png'
        plot_trace_lh(mcmc_res, burn_in=0.4, true_lh=True, fname=fname)

        fname = f'{PLOT_PATH}trace_recall_precision_nz{n_zones}.png'
        plot_trace_recall_precision(mcmc_res, burn_in=0.4, fname=fname)

        # Plot zone size over time
        fname = f'{PLOT_PATH}zone_size_over_time_nz{n_zones}.png'
        plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2, fname=fname)
    """