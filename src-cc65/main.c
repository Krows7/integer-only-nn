// src/main.c
#include <nes.h>
#include <ppu.h> // Include PPU header for PPU control registers

// Define a simple color palette
// (Background, Sprite Palette 0, Sprite Palette 1, Sprite Palette 2)
// Colors are NES color codes (0x00 - 0x3F)
const unsigned char palette[4] = {
    0x0F, // Black background
    0x16, // Blue (for example)
    0x27, // Red (for example)
    0x30  // Grey (for example)
    // Add more palettes (up to 32 bytes total) if needed
};

void main(void) {
    // Turn off rendering before accessing PPU registers
    ppu_off();

    // Load the palette into PPU memory ($3F00)
    // Note: ppu_address sets the high byte, ppu_data writes data and increments
    ppu_address(PPU_PALETTE_BASE); // Set PPU address to $3F00
    ppu_data_fill(palette, sizeof(palette)); // Write palette data

    // Set a background color (using the first color entry $3F00)
    // We already loaded 0x0F there. Let's change it to something else.
    ppu_address(PPU_PALETTE_BASE); // Point to $3F00
    ppu_data(0x21); // Set background color to a light green

    // Turn rendering back on (Enable NMI, background, and sprites)
    // Use PPU_CTRL_NMI | PPU_CTRL_BG_ON | PPU_CTRL_SPR_ON for sprites too
    ppu_on_bg(); // Simplified: Enable NMI and Background rendering only

    // Infinite loop to keep the program running
    while (1) {
        // Wait for VBlank (NMI) - prevents consuming all CPU time
        ppu_wait_nmi();

        // Game logic would go here in a real project
    }
}
