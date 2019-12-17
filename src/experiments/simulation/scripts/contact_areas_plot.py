

if __name__ == '__main__':
    from src.util import load_from, transform_weights_from_log,transform_p_from_log, samples2res
    from src.preprocessing import compute_network, get_sites
    from src.postprocessing import compute_dic
    from src.plotting import plot_posterior_frequency, plot_trace_lh, plot_trace_recall_precision, \
        plot_zone_size_over_time, plot_dics, plot_correlation_weights, plot_histogram_weights, plot_correlation_p, \
        plot_posterior_frequency_family1, plot_posterior_frequency_family2



    PATH = '../../../../' # relative path to contact_zones_directory
    PATH_SIMULATION = f'{PATH}/src/experiments/simulation/'
    PLOT_PATH = f'{PATH}plots/contact_areas/'
    TEST_ZONE_DIRECTORY = 'results/contact_areas/2019-10-21_14-49/'


    # Inheritance and number of runs
    inheritance = 1
    inheritances = [0, 1]
    run = 0

    # general parameters for plots
    ts_posterior_freq = 0.8
    ts_low_frequency = 0.5
    burn_in =  0.2


    for inheritance in inheritances:

        # Load the MCMC results
        sample_path = f'{PATH_SIMULATION}{TEST_ZONE_DIRECTORY}contact_areas_i{inheritance}_{run}.pkl'
        samples = load_from(sample_path)
        mcmc_res = samples2res(samples)
        zones = mcmc_res['zones']

        # Retrieve the sites from the csv and transform into a network
        sites, site_names = get_sites(f'{PATH_SIMULATION}data/sites_simulation.csv')
        network = compute_network(sites)

        # plot posterior frequency including family

        plot_posterior_frequency_family2(
            mcmc_res,
            net = network,
            nz = -1,
            ts_low_frequency = ts_low_frequency,
            ts_posterior_freq = ts_posterior_freq,
            burn_in = burn_in,
            show_zone_bbox = True,
            show_axes = False,
            fname = f'{PLOT_PATH}posterior_frequency_family2_inheritance{inheritance}_run{run}.png'
        )

        """
        # Plot trace of likelihood, recall and precision
        plot_trace_lh(
            mcmc_res,
            burn_in = burn_in,
            true_lh = True,
            fname = f'{PLOT_PATH}trace_likelihood_inheritance{inheritance}_run{run}.png'
        )

        plot_trace_recall_precision(
            mcmc_res,
            burn_in = burn_in,
            fname = f'{PLOT_PATH}trace_recall_precision_inheritance{inheritance}_run{run}.png'
        )

        # Plot zone size over time
        plot_zone_size_over_time(
            mcmc_res,
            r = 0,
            burn_in = burn_in,
            fname = f'{PLOT_PATH}zone_size_over_time_inheritance{inheritance}_run{run}.png'
        )
        """
