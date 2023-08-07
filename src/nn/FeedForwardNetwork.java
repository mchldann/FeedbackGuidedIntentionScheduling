package nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class FeedForwardNetwork
{
	// TODO: Make this dynamic.
	private final int NUM_LAYERS = 3;

	private float[][][] weight;
	private float[][] bias;

	public FeedForwardNetwork(String load_path)
	{
		weight = new float[NUM_LAYERS][][];
		bias = new float[NUM_LAYERS][];

		for (int layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {

			ArrayList<float[]> weights_arr_list = new ArrayList<float[]>();

			// Read weights.
			try (BufferedReader br = new BufferedReader(new FileReader(load_path + "/fc" + (layer_num + 1) + "_w.csv"))) {

				// Skip the first line.
				String line = br.readLine();

				while ((line = br.readLine()) != null) {
					String[] values = line.split(",");
					float[] values_float = new float[values.length - 1];

					// Start at index 1 to omit the row number.
					for (int i = 1; i < values.length; i++) {
						values_float[i - 1] = Float.parseFloat(values[i]);
					}
					weights_arr_list.add(values_float);
				}
			}
			catch (Exception ex)
			{
				System.out.println("Exception encountered: " + ex.getMessage());
				System.exit(0);
			}

			// Read biases.
			bias[layer_num] = new float[weights_arr_list.size()];

			try (BufferedReader br = new BufferedReader(new FileReader(load_path + "/fc" + (layer_num + 1) + "_b.csv"))) {

				// Skip the first line.
				String line = br.readLine();

				int bias_idx = 0;
				while ((line = br.readLine()) != null) {
					String[] values = line.split(",");
					bias[layer_num][bias_idx] = Float.parseFloat(values[1]);
					bias_idx++;
				}
			}
			catch (Exception ex)
			{
				System.out.println("Exception encountered: " + ex.getMessage());
				System.exit(0);
			}

			weight[layer_num] = new float[weights_arr_list.size()][];

			for (int i = 0; i < weight[layer_num].length; i++) {
				weight[layer_num][i] = weights_arr_list.get(i);
			}
		}
	}

	public float[] CalculateOutput(float[] input)
	{
		float[] this_layer_out = null;
		float[] prev_layer_out = input.clone();

		for (int layer_num = 0; layer_num < NUM_LAYERS; layer_num++) {

			this_layer_out = new float[weight[layer_num].length];

			for (int i = 0; i < weight[layer_num].length; i++) {

				this_layer_out[i] = 0.0f;
				for (int j = 0; j < weight[layer_num][i].length; j++) {
					this_layer_out[i] += prev_layer_out[j] * weight[layer_num][i][j];
				}

				// Add bias.
				this_layer_out[i] += bias[layer_num][i];

				// Apply ReLU activation at all layers except the final one.
				if (layer_num < (NUM_LAYERS - 1))
				{
					if (this_layer_out[i] < 0.0f)
					{
						this_layer_out[i] = 0.0f;
					}
				}
			}

			prev_layer_out = this_layer_out.clone();
		}

		return this_layer_out;
	}
}
