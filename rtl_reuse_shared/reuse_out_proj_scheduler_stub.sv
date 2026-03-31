`timescale 1ns/1ps
//---------------------------------------------------------------
// Module: reuse_out_proj_scheduler_stub
// Function:
//   out_proj scheduler using the shared 4x4x4 MAC fabric.
//   This version uses true 16-dim group semantics:
//   - each cycle fetches 4 p_t sub-blocks (16 dims total)
//   - each cycle fetches 4 weight tiles for one 16-dim group
//   - A1/A2/A3 and B1/B2/B3 are delayed by 1/2/3 cycles so that
//     the 4-array pipeline sees one logical group per cycle
//---------------------------------------------------------------
module reuse_out_proj_scheduler_stub #(
    parameter int TILE_SIZE   = 4,
    parameter int DATA_WIDTH  = 16,
    parameter int ACC_WIDTH   = 32,
    parameter int N_BANK      = 6,
    parameter int WDEPTH      = 342,
    parameter int WADDR_W     = $clog2(WDEPTH),
    parameter int DATA_W      = 256,
    parameter int IN_DIM      = 256,
    parameter int OUT_DIM     = 128,
    parameter int P_DEPTH     = IN_DIM / TILE_SIZE,
    parameter int P_ADDR_W    = $clog2(P_DEPTH),
    parameter int Y_DEPTH     = OUT_DIM / TILE_SIZE,
    parameter int Y_ADDR_W    = $clog2(Y_DEPTH)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic start,
    output logic busy,
    output logic done,

    output logic                         p_rd_en,
    output logic [P_ADDR_W-1:0]          p_rd_addr0,
    output logic [P_ADDR_W-1:0]          p_rd_addr1,
    output logic [P_ADDR_W-1:0]          p_rd_addr2,
    output logic [P_ADDR_W-1:0]          p_rd_addr3,
    input  logic signed [DATA_WIDTH-1:0] p_rd_data0 [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] p_rd_data1 [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] p_rd_data2 [TILE_SIZE-1:0],
    input  logic signed [DATA_WIDTH-1:0] p_rd_data3 [TILE_SIZE-1:0],

    output logic                         y_axis_TVALID,
    input  logic                         y_axis_TREADY,
    output logic signed [DATA_WIDTH-1:0] y_axis_TDATA [TILE_SIZE-1:0],

    output logic [1:0]                    fabric_mode,
    output logic [6:0]                    fabric_col_blocks,
    output logic                          fabric_valid_in,
    output logic signed [DATA_WIDTH-1:0]  fabric_A0_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_A1_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_A2_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_A3_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_B0_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_B1_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_B2_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],
    output logic signed [DATA_WIDTH-1:0]  fabric_B3_mat [TILE_SIZE-1:0][TILE_SIZE-1:0],

    input  logic signed [ACC_WIDTH-1:0]   fabric_reduced_vec [TILE_SIZE-1:0],
    input  logic signed [ACC_WIDTH-1:0]   fabric_reduced_mat_0 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [ACC_WIDTH-1:0]   fabric_reduced_mat_1 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [ACC_WIDTH-1:0]   fabric_reduced_mat_2 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic signed [ACC_WIDTH-1:0]   fabric_reduced_mat_3 [TILE_SIZE-1:0][TILE_SIZE-1:0],
    input  logic                          fabric_valid_out
);
    localparam int ROW_GROUPS      = OUT_DIM / (TILE_SIZE * 4); // 8
    localparam int ROWS_PER_GROUP  = 4;
    localparam int ROW_TILES       = OUT_DIM / TILE_SIZE;       // 32
    localparam int K_GROUPS        = IN_DIM / (TILE_SIZE * 4);  // 16
    localparam int PHYS_K_BLOCKS   = IN_DIM / TILE_SIZE;        // 64

    typedef enum logic [2:0] {ST_IDLE, ST_FETCH, ST_DRAIN, ST_WAIT_OUT, ST_WRITE, ST_DONE} st_t;
    st_t st;

    logic [$clog2(ROW_GROUPS)-1:0] row_group_idx;
    logic [1:0]                    row_subtile_idx;
    logic [$clog2(K_GROUPS+1)-1:0] k_group_idx;
    logic [1:0]                    drain_cnt;
    logic [$clog2(ROW_TILES)-1:0]  row_tile_linear;

    logic                          fetch_fire_d1;
    logic [3:0]                    stage_vld;

    logic [3:0][$clog2(N_BANK)-1:0] w_bank_sel;
    logic [3:0][WADDR_W-1:0]        w_addr_sel;
    logic [3:0]                     w_en_sel;
    logic [3:0]                     w_port_sel;
    logic [3:0][DATA_W-1:0]         w_dout_sel;

    logic signed [DATA_WIDTH-1:0] cur_A0 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_A1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_A2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_A3 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_B0 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_B1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_B2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] cur_B3 [TILE_SIZE-1:0][TILE_SIZE-1:0];

    logic signed [DATA_WIDTH-1:0] a0_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] b0_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];

    logic signed [DATA_WIDTH-1:0] a1_d1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] a2_d1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] a2_d2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] a3_d1 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] a3_d2 [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] a3_d3 [TILE_SIZE-1:0][TILE_SIZE-1:0];

    logic signed [DATA_WIDTH-1:0] b1_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] b2_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] b3_reg [TILE_SIZE-1:0][TILE_SIZE-1:0];

    logic                        seen_valid;
    logic signed [ACC_WIDTH-1:0] final_vec [TILE_SIZE-1:0];
    logic                        y_wr_en;
    logic [Y_ADDR_W-1:0]         y_wr_addr;
    logic signed [DATA_WIDTH-1:0] y_wr_data [TILE_SIZE-1:0];

    assign row_tile_linear = row_group_idx * ROWS_PER_GROUP + row_subtile_idx;

    reuse_outproj_weight_sram #(
        .N_BANK (N_BANK),
        .DEPTH  (WDEPTH),
        .ADDR_W (WADDR_W),
        .DATA_W (DATA_W)
    ) u_w_sram (
        .clk(clk),
        .rst_n(rst_n),
        .bank_sel(w_bank_sel),
        .addr_sel(w_addr_sel),
        .en_sel(w_en_sel),
        .port_sel(w_port_sel),
        .dout_sel(w_dout_sel)
    );

    reuse_vec_out_sram #(
        .TILE_SIZE (TILE_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH     (Y_DEPTH),
        .ADDR_W    (Y_ADDR_W)
    ) u_y_sram (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(y_wr_en),
        .wr_addr(y_wr_addr),
        .wr_data(y_wr_data),
        .rd_en(1'b0),
        .rd_addr('0),
        .rd_data(),
        .rd2_en(1'b0),
        .rd2_addr('0),
        .rd2_data()
    );

    assign fabric_mode       = 2'b00;
    assign fabric_col_blocks = K_GROUPS;
    assign busy              = (st != ST_IDLE && st != ST_DONE);
    assign done              = (st == ST_DONE);
    assign p_rd_en           = enable && (st == ST_FETCH) && (k_group_idx < K_GROUPS);
    assign fabric_valid_in   = fetch_fire_d1;

    always_comb begin
        int phys_base_idx;
        int tile_idx0, tile_idx1, tile_idx2, tile_idx3;

        w_bank_sel = '0;
        w_addr_sel = '0;
        w_en_sel   = '0;
        w_port_sel = '0;

        p_rd_addr0 = '0;
        p_rd_addr1 = '0;
        p_rd_addr2 = '0;
        p_rd_addr3 = '0;

        if (enable && st == ST_FETCH && k_group_idx < K_GROUPS) begin
            phys_base_idx = k_group_idx * 4;
            p_rd_addr0 = phys_base_idx + 0;
            p_rd_addr1 = phys_base_idx + 1;
            p_rd_addr2 = phys_base_idx + 2;
            p_rd_addr3 = phys_base_idx + 3;

            tile_idx0 = row_tile_linear * PHYS_K_BLOCKS + phys_base_idx + 0;
            tile_idx1 = row_tile_linear * PHYS_K_BLOCKS + phys_base_idx + 1;
            tile_idx2 = row_tile_linear * PHYS_K_BLOCKS + phys_base_idx + 2;
            tile_idx3 = row_tile_linear * PHYS_K_BLOCKS + phys_base_idx + 3;

            w_bank_sel[0] = tile_idx0 % N_BANK;
            w_addr_sel[0] = tile_idx0 / N_BANK;
            w_bank_sel[1] = tile_idx1 % N_BANK;
            w_addr_sel[1] = tile_idx1 / N_BANK;
            w_bank_sel[2] = tile_idx2 % N_BANK;
            w_addr_sel[2] = tile_idx2 / N_BANK;
            w_bank_sel[3] = tile_idx3 % N_BANK;
            w_addr_sel[3] = tile_idx3 / N_BANK;
            w_en_sel      = 4'b1111;
            w_port_sel    = '0;
        end
    end

    always_comb begin
        cur_A0 = '{default:'0};
        cur_A1 = '{default:'0};
        cur_A2 = '{default:'0};
        cur_A3 = '{default:'0};
        cur_B0 = '{default:'0};
        cur_B1 = '{default:'0};
        cur_B2 = '{default:'0};
        cur_B3 = '{default:'0};

        for (int i = 0; i < TILE_SIZE; i++) begin
            cur_A0[i][0] = w_dout_sel[0][(i*TILE_SIZE+0)*DATA_WIDTH +: DATA_WIDTH];
            cur_A0[i][1] = w_dout_sel[0][(i*TILE_SIZE+1)*DATA_WIDTH +: DATA_WIDTH];
            cur_A0[i][2] = w_dout_sel[0][(i*TILE_SIZE+2)*DATA_WIDTH +: DATA_WIDTH];
            cur_A0[i][3] = w_dout_sel[0][(i*TILE_SIZE+3)*DATA_WIDTH +: DATA_WIDTH];

            cur_A1[i][0] = w_dout_sel[1][(i*TILE_SIZE+0)*DATA_WIDTH +: DATA_WIDTH];
            cur_A1[i][1] = w_dout_sel[1][(i*TILE_SIZE+1)*DATA_WIDTH +: DATA_WIDTH];
            cur_A1[i][2] = w_dout_sel[1][(i*TILE_SIZE+2)*DATA_WIDTH +: DATA_WIDTH];
            cur_A1[i][3] = w_dout_sel[1][(i*TILE_SIZE+3)*DATA_WIDTH +: DATA_WIDTH];

            cur_A2[i][0] = w_dout_sel[2][(i*TILE_SIZE+0)*DATA_WIDTH +: DATA_WIDTH];
            cur_A2[i][1] = w_dout_sel[2][(i*TILE_SIZE+1)*DATA_WIDTH +: DATA_WIDTH];
            cur_A2[i][2] = w_dout_sel[2][(i*TILE_SIZE+2)*DATA_WIDTH +: DATA_WIDTH];
            cur_A2[i][3] = w_dout_sel[2][(i*TILE_SIZE+3)*DATA_WIDTH +: DATA_WIDTH];

            cur_A3[i][0] = w_dout_sel[3][(i*TILE_SIZE+0)*DATA_WIDTH +: DATA_WIDTH];
            cur_A3[i][1] = w_dout_sel[3][(i*TILE_SIZE+1)*DATA_WIDTH +: DATA_WIDTH];
            cur_A3[i][2] = w_dout_sel[3][(i*TILE_SIZE+2)*DATA_WIDTH +: DATA_WIDTH];
            cur_A3[i][3] = w_dout_sel[3][(i*TILE_SIZE+3)*DATA_WIDTH +: DATA_WIDTH];

            for (int j = 0; j < TILE_SIZE; j++) begin
                cur_B0[i][j] = p_rd_data0[j];
                cur_B1[i][j] = p_rd_data1[j];
                cur_B2[i][j] = p_rd_data2[j];
                cur_B3[i][j] = p_rd_data3[j];
            end
        end
    end

    always_comb begin
        fabric_A0_mat = stage_vld[0] ? a0_reg : '{default:'0};
        fabric_B0_mat = stage_vld[0] ? b0_reg : '{default:'0};
        fabric_A1_mat = stage_vld[1] ? a1_d1  : '{default:'0};
        fabric_B1_mat = stage_vld[1] ? b1_reg  : '{default:'0};
        fabric_A2_mat = stage_vld[2] ? a2_d2  : '{default:'0};
        fabric_B2_mat = stage_vld[2] ? b2_reg  : '{default:'0};
        fabric_A3_mat = stage_vld[3] ? a3_d3  : '{default:'0};
        fabric_B3_mat = stage_vld[3] ? b3_reg  : '{default:'0};
    end

    always_comb begin
        for (int i = 0; i < TILE_SIZE; i++)
            y_wr_data[i] = final_vec[i][DATA_WIDTH-1:0];
        y_wr_addr = row_tile_linear[Y_ADDR_W-1:0];
    end

    assign y_wr_en       = (st == ST_WRITE);
    assign y_axis_TVALID = (st == ST_WRITE);
    assign y_axis_TDATA  = y_wr_data;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st              <= ST_IDLE;
            row_group_idx   <= '0;
            row_subtile_idx <= '0;
            k_group_idx     <= '0;
            drain_cnt       <= '0;
            fetch_fire_d1   <= 1'b0;
            stage_vld       <= '0;
            seen_valid      <= 1'b0;
            final_vec       <= '{default:'0};
            a0_reg          <= '{default:'0};
            b0_reg          <= '{default:'0};
            a1_d1           <= '{default:'0};
            a2_d1           <= '{default:'0};
            a2_d2           <= '{default:'0};
            a3_d1           <= '{default:'0};
            a3_d2           <= '{default:'0};
            a3_d3           <= '{default:'0};
            b1_reg          <= '{default:'0};
            b2_reg          <= '{default:'0};
            b3_reg          <= '{default:'0};
        end else begin
            fetch_fire_d1 <= p_rd_en;
            stage_vld[0]  <= fetch_fire_d1;
            stage_vld[1]  <= stage_vld[0];
            stage_vld[2]  <= stage_vld[1];
            stage_vld[3]  <= stage_vld[2];

            if (fetch_fire_d1) begin
                a0_reg <= cur_A0;
                b0_reg <= cur_B0;
                a1_d1 <= cur_A1;
                a2_d1 <= cur_A2;
                a2_d2 <= a2_d1;
                a3_d1 <= cur_A3;
                a3_d2 <= a3_d1;
                a3_d3 <= a3_d2;

                b1_reg <= cur_B1;
                b2_reg <= cur_B2;
                b3_reg <= cur_B3;
            end else if (st == ST_IDLE) begin
                stage_vld <= '0;
                a0_reg <= '{default:'0};
                b0_reg <= '{default:'0};
                a1_d1 <= '{default:'0};
                a2_d1 <= '{default:'0};
                a2_d2 <= '{default:'0};
                a3_d1 <= '{default:'0};
                a3_d2 <= '{default:'0};
                a3_d3 <= '{default:'0};
                b1_reg <= '{default:'0};
                b2_reg <= '{default:'0};
                b3_reg <= '{default:'0};
            end

            if (fabric_valid_out) begin
                seen_valid <= 1'b1;
                for (int i = 0; i < TILE_SIZE; i++)
                    final_vec[i] <= fabric_reduced_vec[i];
            end

            case (st)
                ST_IDLE: begin
                    seen_valid  <= 1'b0;
                    k_group_idx <= '0;
                    drain_cnt   <= '0;
                    if (enable && start) begin
                        row_group_idx   <= '0;
                        row_subtile_idx <= '0;
                        st <= ST_FETCH;
                    end
                end

                ST_FETCH: begin
                    if (k_group_idx < K_GROUPS)
                        k_group_idx <= k_group_idx + 1'b1;
                    else begin
                        drain_cnt <= 2'd3;
                        st <= ST_DRAIN;
                    end
                end

                ST_DRAIN: begin
                    if (drain_cnt == 0)
                        st <= ST_WAIT_OUT;
                    else
                        drain_cnt <= drain_cnt - 1'b1;
                end

                ST_WAIT_OUT: begin
                    if (seen_valid && !fabric_valid_out)
                        st <= ST_WRITE;
                end

                ST_WRITE: begin
                    seen_valid  <= 1'b0;
                    k_group_idx <= '0;
                    if (y_axis_TREADY) begin
                        if (row_group_idx == ROW_GROUPS-1 && row_subtile_idx == ROWS_PER_GROUP-1)
                            st <= ST_DONE;
                        else begin
                            if (row_subtile_idx == ROWS_PER_GROUP-1) begin
                                row_subtile_idx <= '0;
                                row_group_idx   <= row_group_idx + 1'b1;
                            end else begin
                                row_subtile_idx <= row_subtile_idx + 1'b1;
                            end
                            st <= ST_FETCH;
                        end
                    end
                end

                ST_DONE: begin
                    if (!start)
                        st <= ST_IDLE;
                end

                default: st <= ST_IDLE;
            endcase
        end
    end
endmodule
