train_siam_net(siam_net, optimizer,  env, 30000, 6, 150, 4, X_test, y_test, trajectory_len=200, verbose_every=250) #simple 9x9

train_siam_net(siam_net, optimizer,  env, 50000, 7, 100, 4, X_test, y_test, trajectory_len=150) #pillar 9x9

train_siam_net(siam_net, optimizer,  env, 30000, 6, 200, 4, X_test, y_test, trajectory_len=250, verbose_every=250) #snake 7x7