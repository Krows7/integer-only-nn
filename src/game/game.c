#include "game.h"
#include "stdint.h"
#include "neslib.h"

#define PUSH(reg) PUSH_##reg
#define PUSH_x "txa\n" PUSH_a
#define PUSH_y "tya\n" PUSH_a
#define PUSH_a "pha\n"

#define POP(reg) POP_##reg
#define POP_x POP_a "tax\n"
#define POP_y POP_a "tay\n"
#define POP_a "pla\n"

__asm__(".section .nmi_begin_b,\"axG\",@progbits,nmi\n"
    ".weak nmi\n"
    ".globl __default_nmi_b\n"
    "nmi:\n"
    "__default_nmi_b:\n"
    "  php\n"
    "  pha\n"
    "  txa\n"
    "  pha\n"
    "  tya\n"
    "  pha\n"

    ".section .nmi_end_b,\"axG\",@progbits,nmi\n"
    "  pla\n"
    "  tay\n"
    "  pla\n"
    "  tax\n"
    "  pla\n"
    "  plp\n"
    "  rti\n");

// __asm__(
//     ".section .nmi_begin,\"axG\",@progbits,nmi\n"
//     "jsr game_step_before_nmi_\n"
// );

__asm__ (
    ".section .nmi.009,\"ax\",@progbits\n"
    "jsr game_step_before_nmi_\n"

    // // "jsr copy_zero_page_to_buffer\n"
    // // PUSH(a)
    // // PUSH(x)
    // // PUSH(y)
    // "ldx #$00\n"
    // "1:\n"
    // "lda __rc0, x\n"
    // "pha\n"
    // "inx\n"
    // "cpx #$20\n"
    // "bne 1b\n"
    // "jsr game_step_before_nmi\n"
    // // TODO Do something
    // // "jsr copy_buffer_to_zero_page\n"

    // // "ldx #$20\n"
    // // "0:\n"
    // // "dex\n"
    // // "pla\n"
    // // "sta __rc0, x\n"
    // // "cpx #$00\n"
    // // "bne 0b\n"
    // // POP(y)
    // // POP(x)
    // // POP(a)
    // // "jsr apply_nmi\n"
);

__asm__ (
    ".section .nmi.400,\"ax\",@progbits\n"
    "jsr game_step_after_nmi_\n"

    // // "jsr copy_zero_page_to_buffer\n"
    // // PUSH(a)
    // // PUSH(x)
    // // PUSH(y)
    // // "ldx #$00\n"
    // // "1:\n"
    // // "lda $00, x\n"
    // // "pha\n"
    // // "inx\n"
    // // "cpx #$20\n"
    // // "bne 1b\n"

    // "lda #$01\n"
    // "sta VRAM_UPDATE\n"
    // "jsr game_step_after_nmi\n"
    // // "jsr copy_buffer_to_zero_page\n"
    // "ldx #$20\n"
    // "0:\n"
    // "dex\n"
    // "pla\n"
    // "sta $00, x\n"
    // "cpx #$00\n"
    // "bne 0b\n"
    // // POP(y)
    // // POP(x)
    // // POP(a)
    // // "jsr apply_late_nmi\n"
);

// void __attribute__((interrupt)) apply_nmi(void) {
//     game_step_before_nmi();
// }

// void __attribute__((interrupt)) apply_late_nmi(void) {
//     __asm__ volatile (
//         "lda #$01\n"
//         "sta VRAM_UPDATE\n"
//     );
//     game_step_after_nmi();
// }

// extern void neslib_nmi_begin(void);

// void __attribute__((interrupt)) nmi(void) {
//     // 1) Manually clear VBlank:

//     neslib_nmi_begin();

//     // volatile uint8_t dummy = *((volatile uint8_t*)0x2002);

//     // // 2) If you still want NESLIB‐style VRAM updates:
//     // // flush_vram_update(updbuf);

//     // // 3) Perform OAM DMA from your sprite buffer at $0200:
//     // *((volatile uint8_t*)0x2003) = 0x00;      // OAMADDR = 0
//     // *((volatile uint8_t*)0x4014) = 0x02;      // DMA from $0200

//     // 4) Update audio (if using FamiTone):
//     // famitone_update();

//     // 5) Any additional custom work here…
//     //    When this returns, LLVM‐MOS emits RTI (restoring registers & P/V/PC).
// }

// volatile unsigned char rcx_backup[32] __attribute__((section(".zp"), used));
volatile unsigned char rcx_backup[32] __attribute__((section(".zp"), used));
// volatile unsigned char rcx_backup[128] __attribute__((section(".zp"), used));

__attribute__((naked)) void copy_zero_page_to_buffer() {
    __asm__ volatile (
        PUSH(a)
        PUSH(x)
        PUSH(y)

        // -----------------------
        // Save __rcX registers into $80..$9F
        "ldx #$00\n"
        // "push_zp1:\n"
        "1:\n"
        "lda $00, x\n"
        "sta rcx_backup, x\n"
        "inx\n"
        "cpx #$20\n"
        // "cpx #$80\n"
        // "bne push_zp1\n"
        "bne 1b\n"
        // -----------------------

        // "ldx #$00\n"
        // "1:\n"
        // "lda $00, x\n"
        // "pha\n"
        // "inx\n"
        // "cpx #$20\n"
        // "bne 1b\n"
        

        // "ldx #$00\n"
        // "ldy #$00\n"
        // "push_zp:\n"
        // "lda $00,x\n"
        // "sta ($fe),y\n"
        // "inx\n"
        // "iny\n"
        // "cpx #$20\n"
        // "bne push_zp\n"

        POP(y)
        POP(x)
        POP(a)

        "rts\n"
    );
}

__attribute__((naked)) void copy_buffer_to_zero_page() {
    __asm__ volatile (
        PUSH(a)
        PUSH(x)
        PUSH(y)
        
        // -----------------------
        // Restore __rcX registers from $80..$9F
        "ldx #$00\n"
        // "pop_zp1:\n"
        "0:\n"
        "lda rcx_backup, x\n"
        "sta $00, x\n"
        "inx\n"
        "cpx #$20\n"
        // "cpx #$80\n"
        // "bne pop_zp1\n"
        "bne 0b\n"
        // -----------------------

        // "ldx #$20\n"
        // "0:\n"
        // "dex\n"
        // "pla\n"
        // "sta $00, x\n"
        // "cpx #$00\n"
        // "bne 0b\n"
        


        // "ldx #$00\n"
        // "ldy #$00\n"
        // "pop_zp:\n"
        // "lda ($fe),y\n"   // Load from buffer
        // "sta $00,x\n"     // Store to zero page
        // "inx\n"
        // "iny\n"
        // "cpx #$20\n"      // 32 bytes
        // "bne pop_zp\n"

        POP(y)
        POP(x)
        POP(a)

        "rts\n"
    );
}

void game_step_before_nmi_(void) {
    copy_zero_page_to_buffer();
    game_step_before_nmi();
    // TODO Do something
    copy_buffer_to_zero_page();
}

void game_step_after_nmi_(void) {
    copy_zero_page_to_buffer();
    __asm__ volatile (
        "lda #$01\n"
        "sta VRAM_UPDATE\n"
    );
    game_step_after_nmi();
    copy_buffer_to_zero_page();
}