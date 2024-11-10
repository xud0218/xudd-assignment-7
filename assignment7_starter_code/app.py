from flask import Flask, render_template, request, url_for, session
from scipy.stats import t
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    error_term = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error_term

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_


    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Regression Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Fitted Regression Line")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()


    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Store generated data and simulation results in session
    session["X"] = X.tolist()
    session["Y"] = Y.tolist()
    session["slope"] = slope
    session["intercept"] = intercept
    session["slopes"] = slopes  # Store slopes from simulations
    session["intercepts"] = intercepts  # Store intercepts from simulations
    session["N"] = N
    session["S"] = S
    session["beta0"] = beta0
    session["beta1"] = beta1

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.hist(slopes, bins=30, color='gray', edgecolor='black')
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.title("Histogram of Simulated Slopes")
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) > np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) > np.abs(intercept))


    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(session.get("slopes"))
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(session.get("intercepts"))
        observed_stat = intercept
        hypothesized_value = beta0

    p_value = None
    if test_type == "!=":  # Not equal (≠)
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    elif test_type == ">":  # Greater than (>)
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":  # Less than (<)
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        raise ValueError("Invalid test type selected. Choose 'greater', 'less', or 'two-tailed'.")

    fun_message = None
    if p_value <= 0.0001:
        fun_message = "Wow, that p-value is incredibly small!"

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.hist(simulated_stats, bins=30, color='gray', edgecolor='black')
    plt.axvline(observed_stat, color='red', label="Observed Statistic")
    plt.axvline(hypothesized_value, color='blue', linestyle='--', label="Hypothesized Value")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    confidence_level = float(request.form.get("confidence_level"))
    parameter = request.form.get("parameter")  # Either 'slope' or 'intercept'
    
    # Select the correct estimates based on the parameter
    if parameter == "slope":
        estimates = np.array(session.get("slopes", []))
        true_param = session.get("beta1")
    else:
        estimates = np.array(session.get("intercepts", []))
        true_param = session.get("beta0")

    # Clean the estimates to remove any NaN or infinite values
    estimates = estimates[np.isfinite(estimates)]

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)
    
    # Calculate confidence interval for the parameter estimate
    if std_estimate > 0 and S > 1:
        t_critical = t.ppf(1 - (1 - confidence_level / 100) / 2, df=S - 1)
        ci_lower = mean_estimate - t_critical * std_estimate / np.sqrt(S)
        ci_upper = mean_estimate + t_critical * std_estimate / np.sqrt(S)
    else:
        ci_lower, ci_upper = mean_estimate, mean_estimate  # Set bounds to mean if std is zero

    # Check if the true parameter is within the confidence interval
    includes_true = ci_lower <= true_param <= ci_upper if true_param is not None else False

    # Define the path to save the plot
    plot4_path = "static/plot4.png"

    # Plot mean estimate, confidence intervals, and true parameter
    if np.isfinite(ci_lower) and np.isfinite(ci_upper):
        plt.axhline(mean_estimate, color='orange', linestyle='--', label="Mean Estimate")
        plt.axhline(ci_lower, color='blue', linestyle='--', label="Confidence Interval Lower")
        plt.axhline(ci_upper, color='blue', linestyle='--', label="Confidence Interval Upper")

    # Scatter plot of the individual estimates from each simulation
    plt.scatter(range(len(estimates)), estimates, color='gray', label="Estimates")

    # Plot the true parameter if available
    if true_param is not None and np.isfinite(true_param):
        plt.axhline(true_param, color='green', linestyle='--', label="True Parameter")

    # Labeling the plot
    plt.xlabel("Simulation Index")
    plt.ylabel("Estimated Parameter")
    plt.title("Confidence Interval for Parameter Estimate")
    plt.legend()

    # Save and close the plot
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot4=plot4_path,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        parameter=parameter
    )

if __name__ == "__main__":
    app.run(debug=True)
