// Control / dialog IDs. Plain #defines only: consumed by rc.exe on Windows AND
// by swell_resgen.php on macOS/Linux. Keep it ASCII and free of anything but
// #define lines. IDOK / IDCANCEL come from windows.h / swell.h.
#ifndef RAAI_RESOURCE_H
#define RAAI_RESOURCE_H

#define ID_ASSISTANT_DLG   1001
#define ID_OUTPUT_EDIT     1002   // read-only, scrollable conversation log
#define ID_STATUS_TEXT     1003   // single-line status field
#define ID_INPUT_EDIT      1004   // user input
#define ID_SUBMIT_BTN      1005   // "Send"
#define ID_CONFIRM_BTN     1006   // "Confirm" (reserved for Phase 3 mutations)

#endif // RAAI_RESOURCE_H
