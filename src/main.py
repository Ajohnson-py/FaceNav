import multiprocessing
from detection import facial_detection_loop
from app import start_menu_bar_app


def main() -> None:
    not_paused = multiprocessing.Value('b', True)

    # Start face detection process
    face_process = multiprocessing.Process(target=facial_detection_loop, args=(not_paused,))
    face_process.start()

    # Start menu bar process
    menu_process = multiprocessing.Process(target=start_menu_bar_app, args=(not_paused,))
    menu_process.start()

    # Keep running while facial loop and menu bar app is alive
    while face_process.is_alive() and menu_process.is_alive():
        continue

    # If facial loop or menu bar app quits, terminate program
    if face_process.is_alive():
        face_process.terminate()
        face_process.join()
    else:
        menu_process.terminate()
        menu_process.join()


if __name__ == '__main__':
    main()
