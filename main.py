import os
import math
import numpy as np
from flask import Flask, render_template, request
import openai
from dotenv import load_dotenv
import os




# Set OpenAI API key from environment variable


from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_explanation(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500, temperature: float = 0.5) -> str:
    """Generates a natural-language explanation from OpenAI."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert RF-design tutor who explains calculations clearly."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# Check if the API key is loaded correctly (you can remove this line later)
print(f"API Key Loaded: {openai.api_key}")


app = Flask(__name__)



# ---------------------------------------------------------------------------
# Route: Home
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------------------------------
# Route: Question 1  – Wireless Communication System
# ---------------------------------------------------------------------------
# Route for Question 1: Wireless Communication System
@app.route('/question1', methods=['GET', 'POST'])
def question1():
    if request.method == 'POST':
        try:
            # ⬅️ قراءة المدخلات
            signal_bandwidth = float(request.form['signal_bandwidth'])
            unit = request.form['signal_bandwidth_unit']
            if unit == 'kHz':
                signal_bandwidth *= 1e3
            elif unit == 'MHz':
                signal_bandwidth *= 1e6

            quantizer_bits = int(request.form['quantizer_bits'])
            source_encoder = float(request.form['source_encoder'])
            channel_encoder = float(request.form['channel_encoder'])
            interleaver_bits = int(request.form['interleaver_bits'])
            overhead_bits = float(request.form['overhead_bits'])
            voice_duration_ms = float(request.form['speech_duration'])

            # ⬅️ الحسابات
            sampling_rate = 2 * signal_bandwidth  # Nyquist
            quantizer_output = sampling_rate * quantizer_bits / 1e3  # Kbps
            source_output = quantizer_output * source_encoder
            channel_output = source_output / channel_encoder
            interleaver_output = channel_output  # No change
            voice_duration_sec = voice_duration_ms / 1000
            channel_coded_bits = channel_output * 1e3 * voice_duration_sec
            total_bits = channel_coded_bits + overhead_bits
            burst_output = total_bits / voice_duration_sec / 1e3

            # ⬅️ القيم للعرض
            inputs = {
                'Signal Bandwidth': f"{signal_bandwidth:.3f} Hz",
                'Quantizer Bits': quantizer_bits,
                'Source Encoder Compression Rate': source_encoder,
                'Channel Encoder Rate': channel_encoder,
                'Interleaver Bits': interleaver_bits,
                'Overhead Bits': overhead_bits,
                'Voice Segment Duration': f"{voice_duration_ms:.1f} ms"
            }

            results = {
                'Sampling Rate': f"{sampling_rate:.3f} Hz",
                'Quantizer Output Rate': f"{quantizer_output:.3f} Kbps",
                'Source Encoder Output Rate': f"{source_output:.3f} Kbps",
                'Channel Encoder Output Rate': f"{channel_output:.3f} Kbps",
                'Interleaver Output Rate': f"{interleaver_output:.3f} Kbps",
                'Channel‑Coded Bits': f"{channel_coded_bits:.3f} bits",
                'Burst Formatting Output Rate': f"{burst_output:.3f} Kbps"
            }

            # ⬅️ تحضير prompt القصة الرسمية
            story_prompt = (
                f"In a digital voice transmission setup, the signal bandwidth was {inputs['Signal Bandwidth']} with a quantizer of "
                f"{inputs['Quantizer Bits']} bits. The source encoder had a compression rate of {inputs['Source Encoder Compression Rate']}, "
                f"and a channel encoder with a rate of {inputs['Channel Encoder Rate']} was used. "
                f"An interleaver added {inputs['Interleaver Bits']} bits, and overhead bits were {inputs['Overhead Bits']}. "
                f"The system processed voice segments of duration {inputs['Voice Segment Duration']}.\n\n"
                f"Based on this, engineers calculated a sampling rate of {results['Sampling Rate']}, leading to a quantizer output of "
                f"{results['Quantizer Output Rate']}. This was reduced by the source encoder to {results['Source Encoder Output Rate']}, "
                f"and then increased to {results['Channel Encoder Output Rate']} after channel encoding. Interleaving didn’t change the rate. "
                f"Over a {inputs['Voice Segment Duration']} segment, the number of channel-coded bits was {results['Channel‑Coded Bits']}, "
                f"and with overhead, the burst formatting output rate became {results['Burst Formatting Output Rate']}.\n\n"
                f"Talk about the scenario as a short story paragraph with 150 words only. "
                f"Mention the user inputs first, then explain step-by-step how the outputs were calculated in a clear and formal way using natural language."
            )

            # ⬅️ توليد التفسير من GPT
            explanation = generate_explanation(story_prompt)

            return render_template('Question1.html', inputs=inputs, results=results, explanation=explanation)

        except Exception as err:
            return render_template('Question1.html', error=f"Invalid input: {err}")

    return render_template('Question1.html')




# ---------------------------------------------------------------------------
# Route: Question 2  – OFDM System
# ---------------------------------------------------------------------------
 
@app.route("/question2", methods=["GET", "POST"])
def question2():
    if request.method == "POST":
        try:
            # ── Inputs ────────────────────────────────────────────────────
            bandwidth_rb = float(request.form["bandwidth_rb"])
            unit_bw = request.form["bandwidth_rb_unit"]  # Hz / kHz
            if unit_bw == "Hz":
                bandwidth_rb /= 1e3  # convert to kHz

            subcarrier_spacing = float(request.form["subcarrier_spacing"])
            unit_sc = request.form["subcarrier_spacing_unit"]  # Hz / kHz
            if unit_sc == "Hz":
                subcarrier_spacing /= 1e3

            num_ofdm_symbols = int(request.form["num_ofdm_symbols"])
            duration_rb = float(request.form["duration_rb"])
            unit_dur = request.form["duration_rb_unit"]  # ms / seconds
            if unit_dur == "seconds":
                duration_rb *= 1e3  # to ms

            modulation_order = int(request.form["modulation_order"])
            num_parallel_rb = int(request.form["num_parallel_rb"])

            # ── Calculations ─────────────────────────────────────────────
            num_subcarriers = bandwidth_rb / subcarrier_spacing  # per RB
            bits_per_re = np.log2(modulation_order)
            bits_per_symbol = bits_per_re * num_subcarriers
            bits_per_rb = bits_per_symbol * num_ofdm_symbols
            max_tx_rate = (bits_per_rb * num_parallel_rb) / duration_rb  # kbits/ms == Mbit/s

            total_bw_hz = bandwidth_rb * 1e3 * num_parallel_rb
            spectral_eff = (max_tx_rate * 1e3) / total_bw_hz  # bps/Hz

            # ── Display Inputs ───────────────────────────────────────────
            inputs = {
                "Bandwidth per RB": f"{bandwidth_rb:.3f} kHz",
                "Subcarrier Spacing": f"{subcarrier_spacing:.3f} kHz",
                "Number of OFDM Symbols": num_ofdm_symbols,
                "Duration per RB": f"{duration_rb:.3f} ms",
                "Modulation Order": modulation_order,
                "Parallel Resource Blocks": num_parallel_rb,
            }

            results = {
                "Bits per Resource Element": f"{bits_per_re:.3f} bits",
                "Bits per OFDM Symbol": f"{bits_per_symbol:.3f} bits",
                "Bits per Resource Block": f"{bits_per_rb:.3f} bits",
                "Max Transmission Rate": f"{max_tx_rate*1e3:.3f} bit/s",
                "Spectral Efficiency": f"{spectral_eff:.3f} bps/Hz",
            }

            # 🧠 Natural language story-style prompt
            story_prompt = (
                f"In an OFDM system, the bandwidth per resource block was set to {inputs['Bandwidth per RB']} "
                f"with subcarrier spacing of {inputs['Subcarrier Spacing']}. Each block contained "
                f"{inputs['Number of OFDM Symbols']} symbols over {inputs['Duration per RB']}. The modulation used was "
                f"order {inputs['Modulation Order']}, and {inputs['Parallel Resource Blocks']} blocks were used in parallel.\n\n"
                f"Using these inputs, engineers calculated the number of subcarriers per block, then derived the bits carried per symbol. "
                f"Multiplying by the number of symbols gave {results['Bits per Resource Block']} bits per RB. Combining all parallel blocks, "
                f"the total maximum transmission rate was {results['Max Transmission Rate']}. Finally, by dividing this rate over the total bandwidth, "
                f"they found the spectral efficiency to be {results['Spectral Efficiency']}.\n\n"
                f"Write this as a short story of about 150 words that starts by listing the inputs, then explains how the outputs were calculated."
            )

            # ⬅️ Explanation from AI
            explanation = generate_explanation(story_prompt)

            return render_template("Question2.html", inputs=inputs, results=results, explanation=explanation)

        except Exception as err:
            return render_template("Question2.html", error=f"Invalid input: {err}")

    return render_template("Question2.html")





# ---------------------------------------------------------------------------
# Route: Question 3  – Link Budget
# ---------------------------------------------------------------------------
@app.route("/question3", methods=["GET", "POST"])
def question3():
    if request.method == "POST":
        try:
            # ⬅️ المدخلات من الفورم
            path_loss = float(request.form["L_p"])
            transmit_gain = float(request.form["G_t"])
            receive_gain = float(request.form["G_r"])
            data_rate = float(request.form["R"])
            rate_unit = request.form.get("R_unit", "bps")
            if rate_unit == "kbps":
                data_rate *= 1000

            line_loss = float(request.form["L_o"])
            other_losses = float(request.form["L_f"])
            fade_margin = float(request.form["F_margin"])
            tx_amp_gain = float(request.form["A_t"])
            rx_amp_gain = float(request.form["A_r"])
            noise_figure = float(request.form["N_f"])
            temperature_kelvin = float(request.form["T"])
            eb_n0 = float(request.form["SNR_per_bit"])
            link_margin = float(request.form["link_margin"])

            # ⬅️ الثوابت والحسابات
            K_db = -228.6

            Pr_db = (
                link_margin +
                10 * math.log10(data_rate) +
                10 * math.log10(temperature_kelvin) +
                noise_figure +
                K_db +
                eb_n0
            )

            Pt_db = (
                Pr_db +
                path_loss +
                line_loss +
                other_losses +
                fade_margin -
                transmit_gain -
                receive_gain -
                tx_amp_gain -
                rx_amp_gain
            )

            # التحويل إلى واط وتنسيق عشري واضح (بدون e notation)
            Pr_watt = 10 ** (Pr_db / 10)
            Pt_watt = 10 ** (Pt_db / 10)

            Pr_watt_str = f"{Pr_watt:.5f}"
            Pt_watt_str = f"{Pt_watt:.5f}"

            # ⬅️ عرض القيم: Inputs و Results
            inputs = {
                "Path Loss (L_p)": f"{path_loss:.2f} dB",
                "Transmit Antenna Gain (G_t)": f"{transmit_gain:.2f} dB",
                "Receive Antenna Gain (G_r)": f"{receive_gain:.2f} dB",
                "Data Rate (R)": f"{data_rate:.2f} bps",
                "Other Losses (L_o)": f"{line_loss:.2f} dB",
                "Feed Line Loss (L_f)": f"{other_losses:.2f} dB",
                "Fade Margin (F_margin)": f"{fade_margin:.2f} dB",
                "Transmit Amplifier Gain (A_t)": f"{tx_amp_gain:.2f} dB",
                "Receive Amplifier Gain (A_r)": f"{rx_amp_gain:.2f} dB",
                "Noise Figure (N_f)": f"{noise_figure:.2f} dB",
                "Noise Temperature (T)": f"{temperature_kelvin:.2f} K",
                "(E_b/N_0)": f"{eb_n0:.2f} dB",
                "Link Margin": f"{link_margin:.2f} dB",
            }

            results = {
                "Power Received (dB)": f"{Pr_db:.5f} dB",
                "Power Received (Watt)": f"{Pr_watt_str} W",
                "Power Transmitted (dB)": f"{Pt_db:.5f} dB",
                "Power Transmitted (Watt)": f"{Pt_watt_str} W"
            }

            # 🧠 Natural language story-style prompt (like Q2)
            story_prompt = (
                f"In a wireless communication setup, the system experienced a path loss of {inputs['Path Loss (L_p)']}, "
                f"with a transmit antenna gain of {inputs['Transmit Antenna Gain (G_t)']} and a receive antenna gain of {inputs['Receive Antenna Gain (G_r)']}. "
                f"The data rate was {inputs['Data Rate (R)']}, and to ensure link reliability, a link margin of {inputs['Link Margin']} was applied. "
                f"Engineers also considered an Eb/N0 value of {inputs['(E_b/N_0)']}, a receiver noise figure of {inputs['Noise Figure (N_f)']}, "
                f"and a noise temperature of {inputs['Noise Temperature (T)']}. Additional losses included a feed line loss of {inputs['Feed Line Loss (L_f)']} "
                f"and other system losses of {inputs['Other Losses (L_o)']}. Fade margin was set at {inputs['Fade Margin (F_margin)']}, "
                f"while amplifier gains were {inputs['Transmit Amplifier Gain (A_t)']} (Tx) and {inputs['Receive Amplifier Gain (A_r)']} (Rx).\n\n"
                f"Using these inputs, engineers calculated the received signal power by summing the thermal noise contributions, data rate, temperature, noise figure, "
                f"and Eb/N0 along with the Boltzmann constant in dB. This resulted in a required received power of {results['Power Received (dB)']} "
                f"which is equivalent to {results['Power Received (Watt)']}. Then, by adding all losses and subtracting all gains from the received power, "
                f"they computed the total transmit power required: {results['Power Transmitted (dB)']} or {results['Power Transmitted (Watt)']}.\n\n"
                f"Write this as a short story of about 150 words that starts by listing the inputs, then explains how the outputs were calculated."
            )

            # ⬅️ توليد التفسير القصصي
            explanation = generate_explanation(story_prompt)

            return render_template("Question3.html", inputs=inputs, results=results, explanation=explanation)

        except Exception as err:
            return render_template("Question3.html", error=f"Invalid input: {err}")

    return render_template("Question3.html")








# ---------------------------------------------------------------------------
# Route: Question 4  – Cellular System Design
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Route: Question 4  – Cellular System Design
# ---------------------------------------------------------------------------
# Route: Question 4  – Cellular System Design
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Route: Question 4  – Cellular System Design
# ---------------------------------------------------------------------------
@app.route("/question4", methods=["GET", "POST"])
def question4():
    if request.method == "POST":
        try:
            # ⬅️ المدخلات
            area_units = float(request.form["total_area"])
            users = int(request.form["max_num_users"])

            avg_call_duration = float(request.form["avg_call_duration"])
            dur_unit = request.form["avg_call_duration_unit"]
            avg_call_sec = convert_duration_to_seconds(avg_call_duration, dur_unit)

            avg_call_rate = float(request.form["avg_call_rate_per_user"])
            GOS = float(request.form["GOS"])

            SIR = float(request.form["SIR"])
            SIR_unit = request.form["SIR_unit"]
            if SIR_unit == "dB":
                SIR = 10 ** (SIR / 10)

            P0 = float(request.form["P0"])
            P0_unit = request.form["P0_unit"]
            if P0_unit == "dB":
                P0 = db_to_watt(P0)

            receiver_sens = float(request.form["receiver_sensitivity"])
            rs_unit = request.form["receiver_sensitivity_unit"]
            if rs_unit == "dB":
                receiver_sens = db_to_watt(receiver_sens)

            d0 = float(request.form["d0"])
            d0_unit = request.form["d0_unit"]
            d0_m = convert_distance_to_meters(d0, d0_unit)

            path_loss_exp = float(request.form["path_loss_exponent"])
            time_slots = int(request.form["time_slots_per_carrier"])

            # ⬅️ الحسابات
            max_distance = calculate_max_distance(P0, receiver_sens, d0_m, path_loss_exp)
            max_cell_size = calculate_max_cell_size(max_distance)
            total_cells = calculate_total_cells(area_units, max_cell_size)
            traffic_per_user = calculate_traffic_per_user_cps(avg_call_sec, avg_call_rate)
            total_traffic = traffic_per_user * users
            traffic_per_cell = total_traffic / total_cells
            cluster_size = calculate_cluster_size(SIR, path_loss_exp)
            channels_req = calculate_channels_required(traffic_per_cell, GOS)
            carriers_per_cell = calculate_num_carriers_per_cell(channels_req, time_slots)
            carriers_in_system = calculate_num_carriers_in_system(carriers_per_cell, cluster_size)

            # ⬅️ عرض البيانات: Inputs & Results
            inputs = {
                "Total Area": f"{area_units:.1f} sq units",
                "Users": users,
                "Average Call Duration": f"{avg_call_duration:.1f} {dur_unit}",
                "Average Call Rate": f"{avg_call_rate:.1f}",
                "GOS": f"{GOS:.2f}",
                "SIR (linear)": f"{SIR:.3f}",
                "P0": f"{P0:.5f} W",
                "Receiver Sensitivity": f"{receiver_sens:.5f} W",
                "Reference Distance": f"{d0:.1f} {d0_unit}",
                "Path‑Loss Exponent": f"{path_loss_exp:.1f}",
                "Time‑Slots / Carrier": time_slots,
            }

            results = {
                "Max Distance for Reliable Communication": f"{max_distance:.5f} meters",
                "Max Cell Size": f"{max_cell_size:.5f} sq units",
                "Total Number of Cells": int(total_cells),
                "Traffic per User": f"{traffic_per_user:.5f} calls/second",
                "Traffic for All System": f"{total_traffic:.5f} calls/second",
                "Traffic Load for Each Cell": f"{traffic_per_cell:.5f} calls/second",
                "Cluster Size N": cluster_size,
                "Number of Channels Required": channels_req,
                "Number of Carriers per Cell": carriers_per_cell,
                "Number of Carriers in System": carriers_in_system,
            }

            # 🧠 Natural language story-style prompt (like Q2 & Q3)
            story_prompt = (
                f"In a cellular system design, engineers aimed to serve {inputs['Users']} users across an area of {inputs['Total Area']}. "
                f"Each user made approximately {inputs['Average Call Rate']} calls with an average duration of {inputs['Average Call Duration']}. "
                f"To ensure a Grade of Service (GOS) of {inputs['GOS']}, they considered a reference transmit power of {inputs['P0']} "
                f"and receiver sensitivity of {inputs['Receiver Sensitivity']}. A reference distance of {inputs['Reference Distance']} was used, "
                f"and the environment had a path-loss exponent of {inputs['Path‑Loss Exponent']}. The required Signal-to-Interference Ratio (SIR) was "
                f"{inputs['SIR (linear)']}, and each carrier supported {inputs['Time‑Slots / Carrier']} time slots.\n\n"
                f"Based on these inputs, engineers calculated the maximum communication range as {results['Max Distance for Reliable Communication']}, "
                f"resulting in a maximum cell size of {results['Max Cell Size']}. To cover the full area, {results['Total Number of Cells']} cells were needed. "
                f"With a traffic load of {results['Traffic per User']} per user, the system handled a total of {results['Traffic for All System']} calls per second. "
                f"Each cell managed {results['Traffic Load for Each Cell']}, which required {results['Number of Channels Required']} channels. "
                f"A cluster size of {results['Cluster Size N']} led to {results['Number of Carriers in System']} carriers in total for the network.\n\n"
                f"Write this as a short story of about 150 words that starts by listing the inputs, then explains how the outputs were calculated step by step."
            )

            explanation = generate_explanation(story_prompt)

            return render_template("Question4.html", inputs=inputs, results=results, explanation=explanation)

        except Exception as err:
            return render_template("Question4.html", error=f"Invalid input: {err}")

    return render_template("Question4.html")




# 📊  Utility Functions
# ---------------------------------------------------------------------------

CLUSTER_SIZES = [1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 28]

def db_to_watt(db_value: float) -> float:
    return 10 ** (db_value / 10)

def convert_duration_to_seconds(duration: float, unit: str) -> float:
    mapping = {"seconds": 1, "minutes": 60, "hours": 3600}
    return duration * mapping.get(unit, 1)

def convert_distance_to_meters(distance: float, unit: str) -> float:
    return distance * 1_000 if unit == "km" else distance

def calculate_max_distance(P0, receiver_sens, d0, path_loss_exp):
    return d0 * (P0 / receiver_sens) ** (1 / path_loss_exp)

def calculate_max_cell_size(max_distance: float) -> float:
    return 3 * math.sqrt(3) / 2 * (max_distance ** 2)

def calculate_total_cells(total_area: float, max_cell_size: float) -> int:
    return int(np.ceil(total_area / max_cell_size))

def calculate_traffic_per_user_cps(avg_call_sec: float, call_rate: float) -> float:
    return (avg_call_sec * call_rate) / 86400

def calculate_cluster_size(SIR_linear: float, path_loss_exp: float) -> int:
    x = ((SIR_linear * 6) ** (2 / path_loss_exp)) / 3
    for N in CLUSTER_SIZES:
        if N >= x:
            return N
    return CLUSTER_SIZES[-1]

def erlang_b(channels: int, traffic: float) -> float:
    inv_b = 1.0
    for i in range(1, channels + 1):
        inv_b = 1 + (i / traffic) * inv_b
    return 1 / inv_b

def calculate_channels_required(traffic_per_cell: float, GOS: float) -> int:
    erlang_traffic = traffic_per_cell
    for c in range(1, 100):
        if erlang_b(c, erlang_traffic) <= GOS:
            return c
    return 100

def calculate_num_carriers_per_cell(channels: int, time_slots: int) -> float:
    return math.ceil(channels / time_slots)

def calculate_num_carriers_in_system(carriers_per_cell: float, cluster_size: int) -> float:
    return carriers_per_cell * cluster_size

# ---------------------------------------------------------------------------
# 🚀  Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)

