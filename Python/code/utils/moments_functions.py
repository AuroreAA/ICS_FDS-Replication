def expectation_x(mu):
    """Returns non-central moment of order 1"""
    return mu

def expectation_x_2(mu, sigma):
    """Returns non-central moment of order 2"""
    return mu**2 + sigma**2

def expectation_x_3(mu, sigma):
    """Returns non-central moment of order 3"""
    return mu**3 + 3*mu*sigma**2

def expectation_x_4(mu, sigma):
    """Returns non-central moment of order 4"""
    return mu**4 + 6*mu**2*sigma**2 + 3*sigma**4

def expectation_y_2(ind2, mu_y_groups, eps, k):
    """Computes moment of the component of Y y^2"""
    exp = 0
    for i in range(k):
        exp += eps[i] * expectation_x_2(mu=mu_y_groups[i][ind2], sigma=1)
    return exp

def expectation_y_1_1(ind1_a, ind1_b, mu_y_groups, eps, k):
    """Computes joint moment of y_a * y_b"""
    exp = 0
    for i in range(k):
        exp += eps[i] * expectation_x(mu=mu_y_groups[i][ind1_a]) * expectation_x(mu=mu_y_groups[i][ind1_b])
    return exp

def expectation_y_4(ind4, mu_y_groups, eps, k):
    """Computes moment of the component of Y y^4"""
    exp = 0
    for i in range(k):
        exp += eps[i] * expectation_x_4(mu=mu_y_groups[i][ind4], sigma=1)
    return exp

def expectation_y_3_1(ind3, ind1, mu_y_groups, eps, k):
    """Computes joint moment of y_a^3 * y_b"""
    exp = 0
    for i in range(k):
        exp += eps[i] * expectation_x_3(mu=mu_y_groups[i][ind3], sigma=1) * expectation_x(mu=mu_y_groups[i][ind1])
    return exp

def expectation_y_2_2(ind2_a, ind2_b, mu_y_groups, eps, k):
    """Computes joint moment of y_a^2 * y_b^2"""
    exp = 0
    for i in range(k):
        exp += eps[i] * expectation_x_2(mu=mu_y_groups[i][ind2_a], sigma=1) * expectation_x_2(mu=mu_y_groups[i][ind2_b], sigma=1)
    return exp

def expectation_y_2_1_1(ind2, ind1_a, ind1_b, mu_y_groups, eps, k):
    """Computes joint moment of y_c^2 * y_a * y_b"""
    exp = 0
    for i in range(k):
        exp += (eps[i] *
                expectation_x_2(mu=mu_y_groups[i][ind2], sigma=1) *
                expectation_x(mu=mu_y_groups[i][ind1_a]) * expectation_x(mu=mu_y_groups[i][ind1_b]))
    return exp

def expectation_y_1_1_1_1(ind1_a, ind1_b, ind1_c, ind1_d, mu_y_groups, eps, k):
    """Computes joint moment of y_a * y_b * y_c * y_d"""
    exp = 0
    for i in range(k):
        exp += (eps[i] *
                expectation_x(mu=mu_y_groups[i][ind1_a]) * expectation_x(mu=mu_y_groups[i][ind1_b]) *
                expectation_x(mu=mu_y_groups[i][ind1_c]) * expectation_x(mu=mu_y_groups[i][ind1_d]))
    return exp
