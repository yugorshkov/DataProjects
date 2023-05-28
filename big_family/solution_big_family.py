import numpy as np


def free_hours(schedule: str, room_name: str):
    """Функция для преобразования расписания из "Х.ХХ.Х.Х" в {2, 5, 7}
    где {2, 5, 7} - set со свободными часами каждой комнаты
    """
    return room_name, set(np.where(np.array(list(schedule)) == ".")[0].flat)


def make_general_schedule(n: int):
    general_schedule = {}
    for _ in range(n):
        city, room_q = input().split()
        general_schedule.setdefault(city, {})
        for _ in range(int(room_q)):
            timetable, room_name = input().split()
            room_name, timetable = free_hours(timetable, room_name)
            general_schedule[city][room_name] = timetable
    return general_schedule


def find_time_slots(general_schedule: set):
    q, *request = input().split()
    all_city_timetable = []
    for city in request:
        city_shedule = []
        for room in general_schedule[city]:
            city_shedule.append(general_schedule[city][room])
        city_shedule = set.union(*city_shedule)
        all_city_timetable.append(city_shedule)
    hours_for_connect = set.intersection(*all_city_timetable)
    return (hours_for_connect, request)


def find_available_rooms(hours_for_connect: set, request: list):
    if hours_for_connect:
        hour = hours_for_connect.pop()
        rooms = []
        visited = set()
        for city in request:
            for room in general_schedule[city]:
                if hour in general_schedule[city][room]:
                    if city not in visited:
                        rooms.append(room)
                        visited.add(city)
        rooms = " ".join(rooms)
        print(f"Yes {rooms}")
    else:
        print("No")


if __name__ == "__main__":
    n = int(input())
    general_schedule = make_general_schedule(n)

    m = int(input())
    for _ in range(m):
        time_slots, request = find_time_slots(general_schedule)
        find_available_rooms(time_slots, request)
