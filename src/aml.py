from azureml.core import Workspace, Experiment, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.estimator import Estimator

def main():
    # workspace
    ws = Workspace.from_config()

    #compute
    compute = AmlCompute(workspace=ws, name='frodo')

    # datasource
    datastore = Datastore.get(ws, datastore_name='surfrider')

    # experiment
    script_params = {
        "--datastore": datastore.as_mount()
    }

    # Create and run experiment
    estimator = Estimator(source_directory='./',
                            script_params=script_params,
                            compute_target=compute,
                            entry_script='train.py',
                            pip_packages=['opencv-python>=4.1',
                                            'tensorpack==0.9.8',
                                            'tensorflow-gpu>=1.3,<2.0',
                                            'tqdm>=4.36.1',
                                            'cython>=0.29.13',
                                            'scipy>=1.3.1',
                                            'ffmpeg-python',
                                            'wget'])

    
    exp = Experiment(ws, 'surfrider_rcnn')
    run = exp.submit(estimator)

if __name__ == '__main__':
    main()