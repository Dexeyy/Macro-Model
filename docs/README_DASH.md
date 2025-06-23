# Macro Regime Analysis Dashboard

This is a Dash web application for visualizing the results of the macro regime analysis model.

## Features

- Interactive visualization of different economic regime classification methods
- Feature distribution analysis by regime
- Asset performance comparison across different regimes
- Portfolio performance visualization
- Regime transitions timeline
- Ensemble method comparison

## Installation

1. Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

2. Run the data processing pipeline first to generate the necessary data files:

```bash
python main.py
```

3. Start the Dash application:

```bash
python app.py
```

4. Open your web browser and navigate to:

```
http://localhost:8050
```

## Dashboard Components

1. **Regime Classification Methods**: View and select different regime classification methods
2. **Regime Feature Analysis**: Analyze how economic features are distributed across regimes
3. **Asset Performance by Regime**: Compare how different assets perform in each regime
4. **Portfolio Comparison**: Compare equal-weight vs regime-based portfolio performance
5. **Regime Transitions**: Visualize how regimes change over time
6. **Ensemble Comparison**: Compare different classification methods side by side

## Customization

You can customize the dashboard by:

1. Modifying the layout in `app.py`
2. Adding new visualization components
3. Changing the styling in the CSS section

## Deployment

For production deployment, consider using:

- Heroku
- AWS Elastic Beanstalk
- Docker with a reverse proxy like Nginx

## Troubleshooting

If you encounter issues:

1. Check that all data files exist in the `data/processed` directory
2. Verify that you've run the main analysis pipeline successfully
3. Check the console logs for any error messages 