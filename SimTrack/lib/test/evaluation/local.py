from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '../MVTD'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.network_path = '../SimTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '../SimTrack'
    settings.result_plot_path = '../SimTrack/output/test/result_plots'
    settings.results_path = '../SimTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '../SimTrack/Weights'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
