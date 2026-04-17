def simple_hbv(P, T, params):
    # params
    fc = params["fc"]       # field capacity
    beta = params["beta"]   # nonlinearity
    k = params["k"]         # runoff coefficient

    SM = 0
    Q_sim = []

    for t in range(len(P)):
        # recharge
        recharge = (SM / fc) ** beta * P[t]
        
        # update soil moisture
        SM = SM + P[t] - recharge
        SM = min(SM, fc)

        # runoff
        Q = k * recharge
        Q_sim.append(Q)

    return np.array(Q_sim)



