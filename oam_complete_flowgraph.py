#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OAM Complete System - Optical Communication via Orbital Angular Momentum
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from oam_channel_wr import oam_channel_wr
from oam_decoder_wr import oam_decoder_wr
from oam_encoder_wr import oam_encoder_wr
from oam_source_wr import oam_source_wr
from oam_visualizer_wr import oam_visualizer_wr



class oam_complete_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "OAM Complete System - Optical Communication via Orbital Angular Momentum", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("OAM Complete System - Optical Communication via Orbital Angular Momentum")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "oam_complete_flowgraph")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################

        self.oam_visualizer_0 = oam_visualizer_wr(
            enable_dashboard_a=True,
            enable_dashboard_b=True,
            enable_dashboard_c=True,
            enable_dashboard_d=True,
            enable_dashboard_e=True,
            dashboard_step_delay=3.0
        )
        self.oam_source_0 = oam_source_wr(message_text='UIS', symbol_rate=32000, data_rate_multiplier=1.0)
        self.oam_encoder_0 = oam_encoder_wr(
            num_oam_modes=6,
            wavelength=630e-9,
            tx_power=20e-3,
            tx_aperture_size=35e-3,
            grid_size=512
        )
        self.oam_decoder_0 = oam_decoder_wr(
            rx_aperture_size=35e-3,
            grid_size=512
        )
        self.oam_channel_0 = oam_channel_wr(
            propagation_distance=5,
            cn2=(4e-17),
            snr_target=30
        )


        ##################################################
        # Connections
        ##################################################
        self.connect((self.oam_channel_0, 0), (self.oam_decoder_0, 0))
        self.connect((self.oam_decoder_0, 0), (self.oam_visualizer_0, 0))
        self.connect((self.oam_encoder_0, 0), (self.oam_channel_0, 0))
        self.connect((self.oam_source_0, 0), (self.oam_encoder_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "oam_complete_flowgraph")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(top_block_cls=oam_complete_flowgraph, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
