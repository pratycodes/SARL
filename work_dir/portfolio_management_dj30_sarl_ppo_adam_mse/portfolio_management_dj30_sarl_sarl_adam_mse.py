_base_ = [
    '../_base_/datasets/portfolio_management/dj30.py',
    '../_base_/environments/portfolio_management/env.py',
    '../_base_/trainers/portfolio_management/sarl_trainer.py',
    '../_base_/losses/mse.py',
    '../_base_/optimizers/adam.py',
]
data = dict(
    data_path='data/portfolio_management/dj30',
    initial_amount=10000,
    length_day=5,
    tech_indicator_list=[
        'high',
        'low',
        'open',
        'close',
        'adjcp',
        'zopen',
        'zhigh',
        'zlow',
        'zadjcp',
        'zclose',
        'zd_5',
        'zd_10',
        'zd_15',
        'zd_20',
        'zd_25',
        'zd_30',
    ],
    test_dynamic_path='data/portfolio_management/dj30/test_with_label.csv',
    test_path='data/portfolio_management/dj30/test.csv',
    train_path='data/portfolio_management/dj30/train.csv',
    transaction_cost_pct=0.001,
    type='PortfolioManagementDataset',
    valid_path='data/portfolio_management/dj30/valid.csv')
environment = dict(type='PortfolioManagementSARLEnvironment')
trainer = dict(
    agent_name='ppo',
    configs=dict(
        dataset=dict(
            data_path='data/portfolio_management/dj30',
            initial_amount=10000,
            length_day=5,
            tech_indicator_list=[
                'high',
                'low',
                'open',
                'close',
                'adjcp',
                'zopen',
                'zhigh',
                'zlow',
                'zadjcp',
                'zclose',
                'zd_5',
                'zd_10',
                'zd_15',
                'zd_20',
                'zd_25',
                'zd_30',
            ],
            test_dynamic_path=
            'data/portfolio_management/dj30/test_with_label.csv',
            test_path='data/portfolio_management/dj30/test.csv',
            train_path='data/portfolio_management/dj30/train.csv',
            transaction_cost_pct=0.001,
            type='PortfolioManagementDataset',
            valid_path='data/portfolio_management/dj30/valid.csv'),
        gamma=0.99,
        lr=5e-05,
        model=dict(fcnet_activation='relu', fcnet_hiddens=[
            256,
            256,
        ]),
        num_gpus=0,
        num_sgd_iter=30,
        num_workers=0,
        sgd_minibatch_size=128,
        train_batch_size=4000,
        work_dir='work_dir/portfolio_management_dj30_sarl_ppo_adam_mse'),
    epochs=2,
    if_remove=False,
    type='PortfolioManagementSARLTrainer',
    work_dir='work_dir/portfolio_management_dj30_sarl_ppo_adam_mse')
