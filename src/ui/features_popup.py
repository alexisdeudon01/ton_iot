#!/usr/bin/env python3
"""
Features popup GUI - Optional Tkinter popup for displaying common features
"""
import logging

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    from tkinter import scrolledtext
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logger.debug("Tkinter not available - GUI popup disabled")


def show_features_popup(common_features: list, cic_total: int, ton_total: int, non_blocking: bool = True) -> None:
    """
    Show popup with common features found during harmonization
    
    Args:
        common_features: List of common feature names
        cic_total: Total number of CIC-DDoS2019 features
        ton_total: Total number of TON_IoT features
        non_blocking: If True, popup is non-blocking (pipeline continues). Default: True
        
    Returns:
        None (popup is displayed if Tkinter is available)
    """
    if not TKINTER_AVAILABLE:
        logger.debug("Tkinter not available - skipping features popup")
        logger.info(f"Common features found: {len(common_features)} (CIC: {cic_total}, TON: {ton_total})")
        return
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        # Create popup window
        popup = tk.Toplevel(root)
        popup.title("Features Communes Trouvées - Harmonisation")
        popup.geometry("700x500")
        popup.resizable(True, True)
        
        # Title
        title = tk.Label(popup, text="Features Communes entre CIC-DDoS2019 et TON_IoT", 
                       font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Summary
        summary_text = f"Total features CIC-DDoS2019: {cic_total}\n"
        summary_text += f"Total features TON_IoT: {ton_total}\n"
        summary_text += f"Features communes trouvées: {len(common_features)}\n\n"
        summary_label = tk.Label(popup, text=summary_text, font=("Arial", 10))
        summary_label.pack(pady=5)
        
        # Features list
        text_frame = tk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scroll_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=15, width=80)
        scroll_text.pack(fill=tk.BOTH, expand=True)
        
        # Display features
        features_text = "\n".join([f"{i+1}. {feat}" for i, feat in enumerate(common_features)])
        scroll_text.insert(tk.END, features_text)
        scroll_text.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_btn = tk.Button(popup, text="Fermer", command=popup.destroy, 
                            font=("Arial", 10), width=20)
        close_btn.pack(pady=10)
        
        # Make popup non-blocking (don't wait for window to close)
        if non_blocking:
            popup.transient(root)
            popup.lift()
            popup.focus_force()
            
            # Update root to process events without blocking
            root.update()
            
            logger.info(f"   Features popup displayed (non-blocking). Pipeline continuing...")
            # Don't destroy root - let it run in background, user can close popup manually
        else:
            # Blocking mode: wait for user to close popup
            root.mainloop()
            
    except Exception as e:
        logger.warning(f"Could not display features popup: {e}. Continuing...")
