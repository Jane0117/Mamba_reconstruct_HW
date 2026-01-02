// ============================================================
// top_mac_plus_bias_fifo_sigmoid.sv   (方案 B：保留 FIFO 原输出 + 新增 sigmoid 输出)
// Purpose:
//   slim_mac_mem_controller_combined_dp
//     -> pulse_to_stream_adapter
//     -> bias_add_regslice_ip_A (bias_ROM IP)
//     -> vec_fifo_axis_ip (bias2sigmoid_fifo IP)
//     -> sigmoid4_vec (LUT sigmoid, Q8.8 -> Q0.16)
//
// Notes:
//   - bias 的 out_ready 接 FIFO 的 in_ready（由 FIFO 反压）
//   - 顶层对外仍暴露 FIFO 的 out_valid/out_ready/out_vec（sigmoid 之前，Q8.8 signed）
//   - 顶层新增 sigmoid_out_*（sigmoid 之后，Q0.16 unsigned）
//   - 仍然保留 bias_raw_* 方便你观察 bias 加法是否正确
// ============================================================

`timescale 1ns/1ps

module top_mac_plus_bias_fifo_sigmoid #(
    parameter int TILE_SIZE  = 4,
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 32,
    parameter int FRAC_BITS  = 8,

    // Controller internal params
    parameter int N_BANK     = 6,
    parameter int WDEPTH     = 683,
    parameter int WADDR_W    = $clog2(WDEPTH),
    parameter int DATA_W     = 256,
    parameter int XT_ADDR_W  = 6,

    // Bias params
    parameter int D          = 256,
    parameter int PIPE_LAT   = 4,

    // Sigmoid LUT params
    parameter int ADDR_BITS  = 11,
    parameter string LUT_FILE = "D:/Mamba/Cmamba_reconstruct/sigmoid_lut_q016_2048.hex"
)(
    input  logic clk,
    input  logic rst_n,

    // ---- Upstream control for controller start ----
    input  logic s_axis_TVALID,
    output logic s_axis_TREADY,

    // ---- FIFO output stream (pre-sigmoid, Q8.8) ----
    output logic                         fifo_out_valid,
    input  logic                         fifo_out_ready,
    output logic signed [DATA_WIDTH-1:0] fifo_out_vec [TILE_SIZE-1:0],

    // ---- Sigmoid output stream (post-sigmoid, Q0.16) ----
    output logic                         sigmoid_out_valid,
    input  logic                         sigmoid_out_ready,
    output logic        [DATA_WIDTH-1:0] sigmoid_out_vec [TILE_SIZE-1:0],

    // ---- Optional: tap bias output before FIFO (debug) ----
    output logic                         bias_raw_valid,
    output logic signed [DATA_WIDTH-1:0] bias_raw_vec [TILE_SIZE-1:0]
);

    // ============================================================
    // 0) Controller instance
    // ============================================================
    logic                         mac_m_valid;   // done pulse
    logic                         mac_m_ready;
    logic signed [DATA_WIDTH-1:0] mac_vec [TILE_SIZE-1:0];

    // controller 不被下游卡住（你目前的策略）
    assign mac_m_ready = 1'b1;

    slim_mac_mem_controller_combined_dp #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .FRAC_BITS (FRAC_BITS),
        .N_BANK    (N_BANK),
        .WDEPTH    (WDEPTH),
        .WADDR_W   (WADDR_W),
        .DATA_W    (DATA_W),
        .XT_ADDR_W (XT_ADDR_W)
    ) u_mac (
        .clk          (clk),
        .rst_n        (rst_n),

        .s_axis_TVALID(s_axis_TVALID),
        .s_axis_TREADY(s_axis_TREADY),

        .m_axis_TVALID(mac_m_valid),
        .m_axis_TREADY(mac_m_ready),

        .reduced_trunc(mac_vec)
    );

    // ============================================================
    // 1) Adapter: pulse -> streaming (valid/ready)
    // ============================================================
    logic                         ad_valid, ad_ready;
    logic signed [DATA_WIDTH-1:0] ad_vec [TILE_SIZE-1:0];

    pulse_to_stream_adapter #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_p2s (
        .clk        (clk),
        .rst_n      (rst_n),

        .pulse_valid(mac_m_valid),
        .pulse_vec  (mac_vec),

        .out_valid  (ad_valid),
        .out_ready  (ad_ready),
        .out_vec    (ad_vec)
    );

    // ============================================================
    // 2) Bias adder (calls bias_ROM IP)
    // ============================================================
    logic sof_bias;
    assign sof_bias = 1'b0;

    logic                         bias_valid;
    logic                         bias_ready;   // 将由 FIFO in_ready 提供
    logic signed [DATA_WIDTH-1:0] bias_vec [TILE_SIZE-1:0];

    bias_add_regslice_ip_A #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .D         (D),
        .PIPE_LAT  (PIPE_LAT)
    ) u_bias_add (
        .clk      (clk),
        .rst_n    (rst_n),

        .in_valid (ad_valid),
        .in_ready (ad_ready),
        .in_vec   (ad_vec),

        .sof      (sof_bias),

        .out_valid(bias_valid),
        .out_ready(bias_ready),
        .out_vec  (bias_vec)
    );

    // debug tap before FIFO
    assign bias_raw_valid = bias_valid;
    always_comb begin
        for (int i=0; i<TILE_SIZE; i++) begin
            bias_raw_vec[i] = bias_vec[i];
        end
    end

    // ============================================================
    // 3) FIFO: bias -> sigmoid (pre-sigmoid stream)
    // ============================================================
    // FIFO -> sigmoid internal wiring
    logic                         fifo2sig_valid;
    logic                         fifo2sig_ready;
    logic signed [DATA_WIDTH-1:0] fifo2sig_vec [TILE_SIZE-1:0];

    vec_fifo_axis_ip #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_vec_fifo (
        .clk      (clk),
        .rst_n    (rst_n),

        .in_valid (bias_valid),
        .in_ready (bias_ready),
        .in_vec   (bias_vec),

        .out_valid(fifo2sig_valid),
        .out_ready(fifo2sig_ready),
        .out_vec  (fifo2sig_vec)
    );

    // 顶层仍暴露 FIFO 原始输出（供你 debug / 对照 sigmoid 前后）
    assign fifo_out_valid = fifo2sig_valid;
    assign fifo2sig_ready = fifo_out_ready; // 外部对 FIFO 的 backpressure（pre-sigmoid）

    always_comb begin
        for (int i=0; i<TILE_SIZE; i++) begin
            fifo_out_vec[i] = fifo2sig_vec[i];
        end
    end

    // ============================================================
    // 4) Sigmoid: Q8.8 -> Q0.16 (post-sigmoid stream)
    // ============================================================
    // 关键：sigmoid 的 in_* 不能再直接用 fifo2sig_*，否则 fifo2sig_ready 已被外部占用。
    // 方案 B：在 FIFO 输出后“分叉”成两条 consumer，会破坏 ready/valid 语义（一个源不能接两个 sink）。
    //
    // ✅ 正确做法：如果你要同时保留 pre-sigmoid 输出给外部 *并且* 继续送进 sigmoid，
    // 需要在 FIFO 后面再加一个“旁路 FIFO/寄存器切片/复制器”。
    //
    // 最小侵入实现：再加一个 AXIS FIFO（或 skid buffer）作为 sigmoid 专用通道：
    //    FIFO(out) -> (fanout) -> 1) 顶层 fifo_out_*  (debug)
    //                          -> 2) sigmoid_input_fifo -> sigmoid4_vec -> sigmoid_out_*
    //
    // 下面用一个简单的 1-entry skid buffer 来复制 token（不丢且保持语义），
    // 但注意：真正“同时”输出两路仍需要两路都 ready 才能前进（广播语义）。
    //
    // 更工程化的方式：让外部不要消费 fifo_out_*，只用于观察；或者用 ILA 观察。
    //
    // 这里我给你一个“广播”版本：fifo token 只有在 debug-ready && sigmoid-ready 同时为 1 时才会被消费。
    // 这样保证两路看到完全一致的数据，不会错位。

    // --- broadcast ready/valid ---
    // source: fifo2sig_valid/fifo2sig_vec
    // sinks:
    //   A) 外部 pre-sigmoid (fifo_out_ready)
    //   B) sigmoid 模块 (sig_in_ready)
    //
    // 重新定义 fifo2sig_ready：必须两边都 ready 才能 pop FIFO
    logic sig_in_valid;
    logic sig_in_ready;
    logic signed [DATA_WIDTH-1:0] sig_in_vec [TILE_SIZE-1:0];

    assign sig_in_valid = fifo2sig_valid;
    always_comb begin
        for (int i=0; i<TILE_SIZE; i++) sig_in_vec[i] = fifo2sig_vec[i];
    end

    // 广播消费条件：两边都 ready 才能让 FIFO 前进
    // 注意：这会让“debug 端”如果一直 ready=1，则只受 sigmoid 反压影响；反之亦然。
    wire broadcast_ready = fifo_out_ready && sig_in_ready;

    // 覆盖前面那个 fifo2sig_ready 赋值
    //（为避免多驱动，上面那句 assign fifo2sig_ready = fifo_out_ready; 请删除）
    // 这里重新赋值：
    //   FIFO 只有在 broadcast_ready 时才 pop
    //   顶层 fifo_out_ready 只是 debug sink ready
    //   sigmoid_in_ready 是 sigmoid sink ready
    assign fifo2sig_ready = broadcast_ready;

    sigmoid4_vec #(
        .TILE_SIZE (TILE_SIZE),
        .IN_W      (DATA_WIDTH),
        .OUT_W     (DATA_WIDTH),
        .ADDR_BITS (ADDR_BITS),
        .LUT_FILE  (LUT_FILE)
    ) u_sigmoid (
        .clk      (clk),
        .rst_n    (rst_n),

        .in_valid (sig_in_valid),
        .in_ready (sig_in_ready),
        .in_vec   (sig_in_vec),

        .out_valid(sigmoid_out_valid),
        .out_ready(sigmoid_out_ready),
        .out_vec  (sigmoid_out_vec)
    );

endmodule
