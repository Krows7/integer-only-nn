#ifndef __NES__ 
#define __NES__
#endif
#ifdef __NES__

#include "neslib.h"
#include <ines.h>
#include "nes.h"
#include "base.h"

#define asm __asm__

MAPPER_PRG_ROM_KB(32);
MAPPER_PRG_RAM_KB(8);
MAPPER_CHR_ROM_KB(8);

static const char msg[] = "HELLO WORLD";

// helper: print a C-string at (x,y) on the BG nametable
void ppu_puts(unsigned x, unsigned y, const char *s) {
    // compute NT address (nametable A)
    unsigned addr = NTADR_A(x,y);
    // set VRAM write address
    // PPU_ADDR = addr >> 8;
    // PPU_ADDR = addr & 0xFF;
    vram_adr(addr);
    // write each character’s tile index
    while (*s) {
        // if your .chr maps ASCII space (0x20) → tile 0:
        //   tile_index = *s - 0x20;
        // adjust this if your font is laid out differently
        unsigned char tile = *s - 0x20;
        // PPU_DATA = tile;
        vram_put(tile);
        ++s;
    }
}

#define VBLANK_FLAG 0x80

char x, y = 50;

// extern void _nmi(void);

// extern void __default_nmi(void);

extern void nmi(void);

volatile char nmi_counter = 0;

// __attribute__((interrupt))
// void __default_nmi(void) {
// __attribute__((section(".nmi.10000"))) void nmi_test(void) {
// __attribute__((section(".nmi.060"))) void custom_nmi_tasks(void) {
// __attribute__((section(".nmi.080"))) __attribute__((noinline))
// void custom_nmi_tasks(void) __asm__("__bank_nmi");
void custom_nmi_tasks(void) {
// __attribute__((interrupt)) void nmi(void) {
// void nmi(void) {
    // _nmi();
    // nmi();
    // __default_nmi();

    // ++nmi_counter;

    println("%d", (PPU.status & VBLANK_FLAG) != 0);
    println("huy");
    println("%d", (PPU.status & VBLANK_FLAG) != 0);

    // oam_clear();

    // const char pad_state = pad_poll(0);
    // const char speed = 1;

    // if (pad_state & PAD_UP) y -= speed;
    // else if (pad_state & PAD_DOWN) y += speed;

    // if (pad_state & PAD_LEFT) x -= speed;
    // else if (pad_state & PAD_RIGHT) x += speed;

    // oam_spr(x, y, 'e', 0);
}

// extern void neslib_nmi_begin(void);
extern void __default_nmi(void);

// asm (
//     ".globl nmi\n"
//     "nmi:\n\t\t"
//     "jsr huyy\n\t"
//     // "jmp __default_nmi\n\t"
//     "jmp neslib_nmi_begin\n\t"
// );

asm (
    ".section .nmi.009,\"ax\",@progbits\n"
    "jsr huyy\n"
);

// asm (
//     ".globl __default_nmi\n"
//     "__default_nmi:\n"
//     "  jsr huyy\n"
//     "  jmp nmi\n"
// );

// asm(".section .nmi_end,\"axG\",@progbits,nmi\n"
//     // "  pla\n"
//     // "  tay\n"
//     // "  pla\n"
//     // "  tax\n"
//     // "  pla\n"
//     "  jsr huyy\n"
//     "  rti\n");

uint8_t color = 0;

void huyy(void) {
    // println("LLL");
    ++color;
    // if (++color % 30 == 0) {
        PPU.vram.address = 0x3f00 >> 8;
        PPU.vram.address = 0x3f00 & 0xff;
        // ppu_write_addr(0x3f00);
        PPU.vram.data = color;

        PPU.vram.address = 0;
        PPU.vram.address = 0;

        // PPU.scroll = 0;
        // PPU.scroll = 0;

        // PPU.vram.address = 0x3f00 >> 8;
        // PPU.vram.address = 0x3f00 & 0xff;
        // PPU.vram.data = 0;

        // char a = '0' + (color & 7);
        // vram_adr(NTADR_A(10, 10));
        // // vram_write(&a, 1);
        // vram_put(a);
        // println("%d", color);
    // }
}

// __attribute__((interrupt))
// void nmi(void) {
// // void irq(void) {
//     // println("GGG");
//     // asm volatile ("jmp %0" :: "i"(neslib_nmi_begin));
//     __asm__ volatile ("jmp %0" :: "i"(__default_nmi));
// }

// __attribute__((interrupt)) __attribute((noinline))
// void irq(void) {
// // void irq(void) {
//     println("YYY");
//     // asm volatile ("jmp %0" :: "i"(neslib_nmi_begin));
// }

// __asm__ (
//     // ".include \"/home/huy/llvm-mos-sdk/mos-platform/nes-mmc3/irq.s\"\n\t"
//     // ".section .text.bank_nmi,\"ax\",@progbits\n\t"
//     // ".weak bank_nmi\n\t"
//     // ".globl __bank_nmi\n\t"
//     // "__bank_nmi:\n\t"
//     ".section .init.987,\"ax\",@progbits\n\t"
//     // ".weak bank_nmi\n\t"
//     ".globl huy\n\t"
//     "huy:\n\t\t"
//     "jsr custom_nmi_tasks\n\t"
//     // "rts"
// );

const char pal[] = {0x0f, 0x10, 0x20, 0x30};

void huy(void) {
    println("AAA");
}

int main(void) {
    // custom_nmi_tasks();
    // turn off rendering while we set things up
    ppu_off();

    // // select CHR bank 0 for background tiles
    bank_bg(0);

    pal_bg(pal);

    vram_adr(NAMETABLE_A);
    vram_fill(' ', 32 * 30);

    vram_adr(NTADR_A(10, 10));
    vram_write(msg, sizeof(msg) - 1);

    // // turn on both BG and SPR
    ppu_on_all();

    // set_irq_ptr(huyy);

    // // wait a frame so the PPU is fully up
    // ppu_wait_frame();

    // // write “HELLO WORLD” at tile-coords (5, 13)
    // ppu_puts(5, 13, msg);

    // loop forever
    for (;;) {
        ppu_wait_frame();
        // pal_col(0, nmi_counter);
        // println("%d", nmi_counter);
    }

    return 0;
}

#endif